"""Tests for session protection around the analysis API."""

from app import create_app


def test_current_user_reports_anonymous_sessions():
    app = create_app()
    app.config["TESTING"] = True

    response = app.test_client().get("/api/auth/me")

    assert response.status_code == 401
    assert response.get_json() == {"authenticated": False}


def test_analysis_api_requires_an_authenticated_session():
    app = create_app()
    app.config["TESTING"] = True

    response = app.test_client().post("/api/analysis/analyze")

    assert response.status_code == 401
    assert response.get_json()["error"] == "Authentication is required"


def test_profile_returns_safe_identity_and_zero_activity_for_new_user():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["user"] = {"id": "profile-user", "name": "Profile User", "email": "profile@example.com"}

    response = client.get("/api/auth/profile")

    assert response.status_code == 200
    assert response.get_json()["user"]["email"] == "profile@example.com"
    assert response.get_json()["stats"]["images_analyzed"] == 0
