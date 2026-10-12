"""Tests for session protection around the analysis API."""

import uuid

from app import create_app


def test_current_user_reports_anonymous_sessions():
    app = create_app()
    app.config["TESTING"] = True
    app.config["EMAIL_VERIFICATION_REQUIRED"] = False

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


def test_local_registration_and_password_login_create_a_safe_session():
    app = create_app()
    app.config["TESTING"] = True
    app.config["EMAIL_VERIFICATION_REQUIRED"] = False
    client = app.test_client()
    email = f"student-{uuid.uuid4().hex}@example.com"

    registration = client.post(
        "/api/auth/register",
        json={"name": "Local Student", "email": email, "password": "safe-password-123"},
    )

    assert registration.status_code == 201
    assert registration.get_json()["user"]["email"] == email
    client.post("/api/auth/logout")

    login = client.post("/api/auth/login", json={"email": email, "password": "safe-password-123"})

    assert login.status_code == 200
    assert login.get_json()["authenticated"] is True


def test_local_login_does_not_expose_account_existence():
    app = create_app()
    app.config["TESTING"] = True

    response = app.test_client().post(
        "/api/auth/login",
        json={"email": f"missing-{uuid.uuid4().hex}@example.com", "password": "safe-password-123"},
    )

    assert response.status_code == 401
    assert response.get_json()["error"] == "Invalid email or password"
