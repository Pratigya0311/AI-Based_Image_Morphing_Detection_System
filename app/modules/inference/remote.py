"""Adapter for running the model pipeline in a Hugging Face Gradio Space."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file

from app.modules.results import AnalysisResult, ResultService


class HuggingFaceSpacePipeline:
    """Call a Gradio Space for inference and preserve the local result contract."""

    def __init__(self, space_id: str) -> None:
        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is required when HF_INFERENCE_SPACE is configured")
        self.client = Client(space_id, token=token)
        self.results = ResultService()

    def analyze_image(self, submission_id: str, image_bytes: bytes) -> AnalysisResult:
        """Send one acquired image to the Space and return the app result type."""
        suffix = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix) as image_file:
            image_file.write(image_bytes)
            image_file.flush()
            response: Any = self.client.predict(
                handle_file(image_file.name), submission_id,
                api_name="/analyze",
            )

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Inference Space returned an invalid response") from exc
        if not isinstance(response, dict):
            raise RuntimeError("Inference Space returned an invalid response")
        probabilities = response.get("class_probabilities")
        if not isinstance(probabilities, dict):
            raise RuntimeError("Inference Space response is missing class probabilities")
        return self.results.create_result(submission_id, probabilities)
