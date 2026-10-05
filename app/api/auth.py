"""Google OpenID Connect authentication for the protected analysis APIs."""

from __future__ import annotations

from functools import wraps

from authlib.integrations.flask_client import OAuth
from flask import Blueprint, current_app, jsonify, redirect, session

from app.core.database import database

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
oauth = OAuth()


def init_auth(app) -> None:
    """Configure the OAuth client using deployment environment variables."""
    oauth.init_app(app)
    if not app.config.get("GOOGLE_CLIENT_ID") or not app.config.get("GOOGLE_CLIENT_SECRET"):
        return
    oauth.register(
        name="google",
        client_id=app.config["GOOGLE_CLIENT_ID"],
        client_secret=app.config["GOOGLE_CLIENT_SECRET"],
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def login_required(view):
    """Require a local session created only after a verified Google callback."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Authentication is required"}), 401
        return view(*args, **kwargs)

    return wrapped


@auth_bp.route("/google", methods=["GET"])
def google_login():
    """Start the Google authorization-code flow."""
    if not current_app.config.get("GOOGLE_OAUTH_ENABLED"):
        return jsonify({"error": "Google sign-in is not configured on this server"}), 503
    # Requests originate through Vite's /api proxy in development, so relying
    # on the request Host can incorrectly produce localhost:5173 here. Google
    # requires this value to exactly match the registered backend callback URI.
    callback_url = f"{current_app.config['BACKEND_URL'].rstrip('/')}/api/auth/google/callback"
    return oauth.google.authorize_redirect(callback_url)


@auth_bp.route("/google/callback", methods=["GET"])
def google_callback():
    """Validate Google identity data and create a minimal local session."""
    token = oauth.google.authorize_access_token()
    profile = token.get("userinfo")
    if not profile or not profile.get("email_verified"):
        return jsonify({"error": "Google did not return a verified email address"}), 403

    user = database.upsert_google_user(profile)
    session.clear()
    session.permanent = True
    session["user"] = {
        "id": user.id,
        "name": user.display_name,
        "email": user.email,
        "picture": user.picture_url,
    }
    # Tokens are intentionally not stored in the browser session; this project
    # only needs identity, not access to a user's Google resources.
    return redirect(current_app.config["FRONTEND_URL"])


@auth_bp.route("/me", methods=["GET"])
def current_user():
    """Return the safe identity fields used by the frontend."""
    user = session.get("user")
    if user is None:
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "user": user})


@auth_bp.route("/profile", methods=["GET"])
@login_required
def profile():
    """Return the signed-in profile with durable personal activity totals."""
    user = session["user"]
    return jsonify(
        {
            "user": user,
            "stats": database.profile_stats(user["email"]),
        }
    )


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """End the local application session."""
    session.clear()
    return jsonify({"authenticated": False})
