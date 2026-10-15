"""Google and local-account authentication for protected analysis APIs."""

from __future__ import annotations

from functools import wraps
from email.message import EmailMessage
import re
import smtplib
from urllib.parse import quote

from authlib.integrations.flask_client import OAuth
from flask import Blueprint, current_app, jsonify, redirect, request, session
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash
import requests

from app.core.database import database
from app.modules.audit import AuditLogger

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
oauth = OAuth()
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


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
    """Require a verified local or Google-backed application session."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Authentication is required"}), 401
        return view(*args, **kwargs)

    return wrapped


def _session_user(user) -> dict:
    """Expose only the identity fields required by the browser."""
    return {
        "id": user.id,
        "name": user.display_name,
        "email": user.email,
        "picture": user.picture_url,
        "role": user.role,
    }


def _sign_in(user) -> None:
    session.clear()
    session.permanent = True
    session["user"] = _session_user(user)
    AuditLogger().record("user_signed_in", "success", actor_email=user.email)


def admin_required(view):
    """Require a currently valid deployment-managed administrator role.

    The browser session is intentionally small and can outlive a role change
    made during application start-up (for example when ``ADMIN_EMAIL`` is
    configured after the account already exists).  Re-read the role from the
    database here so an old ``user`` session is promoted correctly and a
    removed admin role cannot continue to access protected endpoints.
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        session_user = session.get("user")
        account = database.account_details(session_user.get("email")) if session_user else None
        if not account or account.get("role") != "admin":
            return jsonify({"error": "Administrator access is required"}), 403
        session_user["role"] = "admin"
        session["user"] = session_user
        return view(*args, **kwargs)
    return wrapped


def _credentials(payload: dict, include_name: bool = False):
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    name = str(payload.get("name", "")).strip()
    if not EMAIL_PATTERN.fullmatch(email):
        return None, None, None, ("Enter a valid email address", 400)
    if len(password) < 8 or len(password) > 128:
        return None, None, None, ("Password must contain 8 to 128 characters", 400)
    if include_name and not 2 <= len(name) <= 255:
        return None, None, None, ("Enter your name using 2 to 255 characters", 400)
    return name, email, password, None


def _valid_new_password(password: object):
    value = str(password or "")
    if not 8 <= len(value) <= 128:
        return None, ("Password must contain 8 to 128 characters", 400)
    return value, None


def _reset_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="password-reset-v1")


def _send_email(email: str, subject: str, content: str) -> None:
    """Send transactional mail through Apps Script, Resend, or local SMTP.

    Google Apps Script and Resend use HTTPS, so they work on hosts that block
    SMTP ports. SMTP remains available as a local-development fallback.
    """
    apps_script_url = current_app.config.get("GOOGLE_APPS_SCRIPT_URL")
    apps_script_token = current_app.config.get("GOOGLE_APPS_SCRIPT_TOKEN")
    if apps_script_url or apps_script_token:
        if not (apps_script_url and apps_script_token):
            raise RuntimeError(
                "GOOGLE_APPS_SCRIPT_URL and GOOGLE_APPS_SCRIPT_TOKEN must both be configured"
            )
        try:
            response = requests.post(
                apps_script_url,
                json={
                    "token": apps_script_token,
                    "to": email,
                    "subject": subject,
                    "text": content,
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError("Google Apps Script email request failed") from exc
        if payload.get("ok") is not True:
            raise RuntimeError("Google Apps Script declined the email request")
        return

    resend_key = current_app.config.get("RESEND_API_KEY")
    resend_from = current_app.config.get("RESEND_FROM")
    if resend_key:
        if not resend_from:
            raise RuntimeError("RESEND_FROM must be configured with RESEND_API_KEY")
        try:
            response = requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_key}"},
                json={"from": resend_from, "to": [email], "subject": subject, "text": content},
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError("Transactional email API request failed") from exc
        return

    required = ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM")
    if not all(current_app.config.get(item) for item in required):
        raise RuntimeError("Email delivery is not configured")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = current_app.config["SMTP_FROM"]
    message["To"] = email
    message.set_content(content)
    with smtplib.SMTP(current_app.config["SMTP_HOST"], current_app.config["SMTP_PORT"], timeout=15) as server:
        server.starttls()
        server.login(current_app.config["SMTP_USERNAME"], current_app.config["SMTP_PASSWORD"])
        server.send_message(message)


def _email_delivery_configured() -> bool:
    """Return whether either supported transactional-email provider is ready."""
    if current_app.config.get("GOOGLE_APPS_SCRIPT_URL") and current_app.config.get(
        "GOOGLE_APPS_SCRIPT_TOKEN"
    ):
        return True
    if current_app.config.get("RESEND_API_KEY") and current_app.config.get("RESEND_FROM"):
        return True
    return all(
        current_app.config.get(item)
        for item in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM")
    )


def _send_password_reset(email: str, token: str) -> bool:
    """Send an expiring reset link through the configured transactional email provider."""
    reset_url = f"{current_app.config['FRONTEND_URL'].rstrip('/')}?reset_token={quote(token)}"
    _send_email(
        email,
        "Reset your AI Morph Detection password",
        "A password reset was requested for your AI Morph Detection account.\n\n"
        f"Use this link within one hour: {reset_url}\n\n"
        "If you did not request this, you can ignore this email.",
    )
    return True


def _send_verification_email(email: str, token: str) -> None:
    """Email a one-hour account-verification link before a local account exists."""
    verify_url = f"{current_app.config['FRONTEND_URL'].rstrip('/')}?verify_token={quote(token)}"
    _send_email(
        email,
        "Verify your AI Morph Detection account",
        "Verify your email to activate your AI Morph Detection account.\n\n"
        f"Use this link within one hour: {verify_url}\n\n"
        "If you did not create this account, you can ignore this email.",
    )


@auth_bp.route("/register", methods=["POST"])
def register():
    """Start verified registration; accounts activate only after email approval."""
    payload = request.get_json(silent=True) or {}
    name, email, password, error = _credentials(payload, include_name=True)
    if error:
        return jsonify({"error": error[0]}), error[1]
    if database.local_user_by_email(email) is not None:
        return jsonify({"error": "An account already exists for this email address"}), 409
    if not current_app.config["EMAIL_VERIFICATION_REQUIRED"]:
        user = database.create_local_user(name, email, generate_password_hash(password))
        assert user is not None
        AuditLogger().record("account_created", "success", actor_email=user.email)
        _sign_in(user)
        return jsonify({"authenticated": True, "user": _session_user(user)}), 201
    token = database.create_pending_registration(name, email, generate_password_hash(password))
    try:
        _send_verification_email(email, token)
    except (OSError, RuntimeError):
        current_app.logger.exception("Verification email could not be sent")
        return jsonify({"error": "Email verification is not configured or could not send a message"}), 503
    return jsonify({"message": "Check your email to verify and activate your account."}), 202


@auth_bp.route("/verify-email", methods=["POST"])
def verify_email():
    """Activate a pending registration from its signed, one-hour email link."""
    token = str((request.get_json(silent=True) or {}).get("token", ""))
    user = database.consume_pending_registration(token)
    if user is None:
        return jsonify({"error": "This verification link is invalid, expired, or already used"}), 400
    AuditLogger().record("account_created", "success", actor_email=user.email)
    _sign_in(user)
    return jsonify({"authenticated": True, "user": _session_user(user)}), 201


@auth_bp.route("/login", methods=["POST"])
def password_login():
    """Authenticate an existing local account without exposing account existence."""
    payload = request.get_json(silent=True) or {}
    _, email, password, error = _credentials(payload)
    if error:
        return jsonify({"error": error[0]}), error[1]
    user = database.local_user_by_email(email)
    if user is None or not user.password_hash or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401
    database.mark_logged_in(user.id)
    _sign_in(user)
    return jsonify({"authenticated": True, "user": _session_user(user)})


@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Email a one-hour reset token without revealing account existence."""
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    if not EMAIL_PATTERN.fullmatch(email):
        return jsonify({"error": "Enter a valid email address"}), 400
    if not _email_delivery_configured():
        return jsonify({"error": "Password reset email is not configured on this server"}), 503
    user = database.local_user_by_email(email)
    if user is not None and user.password_hash:
        token = _reset_serializer().dumps({"email": user.email, "user_id": user.id})
        try:
            _send_password_reset(user.email, token)
        except (OSError, RuntimeError):
            current_app.logger.exception("Password reset email could not be sent")
            return jsonify({"error": "Password reset email could not be sent. Please try again later."}), 503
    return jsonify({"message": "If an account exists for this email, a reset link has been sent."}), 202


@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    """Accept a valid, short-lived email reset token and replace the password."""
    payload = request.get_json(silent=True) or {}
    password, error = _valid_new_password(payload.get("password"))
    if error:
        return jsonify({"error": error[0]}), error[1]
    try:
        reset_data = _reset_serializer().loads(str(payload.get("token", "")), max_age=3600)
    except (BadSignature, SignatureExpired):
        return jsonify({"error": "This password reset link is invalid or has expired"}), 400
    user = database.local_user_by_email(reset_data.get("email"))
    if user is None or user.id != reset_data.get("user_id"):
        return jsonify({"error": "This password reset link is invalid or has expired"}), 400
    database.update_password(user.id, generate_password_hash(password))
    AuditLogger().record("password_reset", "success", actor_email=user.email)
    return jsonify({"message": "Password reset successfully. You can now sign in."})


@auth_bp.route("/change-password", methods=["POST"])
@login_required
def change_password():
    """Allow a signed-in local user to safely choose a new password."""
    payload = request.get_json(silent=True) or {}
    password, error = _valid_new_password(payload.get("new_password"))
    if error:
        return jsonify({"error": error[0]}), error[1]
    user = database.local_user_by_email(session["user"]["email"])
    if user is None or not user.password_hash:
        return jsonify({"error": "This account uses Google sign-in and has no local password"}), 400
    if not check_password_hash(user.password_hash, str(payload.get("current_password", ""))):
        return jsonify({"error": "Current password is incorrect"}), 401
    database.update_password(user.id, generate_password_hash(password))
    AuditLogger().record("password_changed", "success", actor_email=user.email)
    return jsonify({"message": "Password updated successfully"})


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

    existing_user = database.local_user_by_email(profile["email"])
    user = database.upsert_google_user(profile)
    if existing_user is None:
        AuditLogger().record("account_created", "success", actor_email=user.email)
    _sign_in(user)
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
    account = database.account_details(user["email"])
    return jsonify({"user": account or user, "stats": database.profile_stats(user["email"])})


@auth_bp.route("/admin/users", methods=["GET"])
@login_required
@admin_required
def admin_users():
    """List accounts for the protected administrator dashboard."""
    return jsonify({"users": database.list_users()})


@auth_bp.route("/admin/activity", methods=["GET"])
@login_required
@admin_required
def admin_activity():
    """Return a page of operational audit events for the admin dashboard."""
    limit = min(max(request.args.get("limit", default=12, type=int) or 12, 1), 100)
    offset = max(request.args.get("offset", default=0, type=int) or 0, 0)
    events = AuditLogger().list_events(limit=limit + 1, offset=offset, newest_first=True)
    has_more = len(events) > limit
    return jsonify({
        "events": [event.to_dict() for event in events[:limit]],
        "has_more": has_more,
    })


@auth_bp.route("/admin/users/<user_id>", methods=["DELETE"])
@login_required
@admin_required
def admin_delete_user(user_id: str):
    """Delete a non-admin user and their personal history by explicit admin action."""
    if user_id == session["user"]["id"]:
        return jsonify({"error": "An administrator cannot delete their own account"}), 400
    target = next((user for user in database.list_users() if user["id"] == user_id), None)
    if target is None:
        return jsonify({"error": "User not found"}), 404
    if target["role"] == "admin":
        return jsonify({"error": "Administrator accounts cannot be removed here"}), 403
    database.delete_user(user_id)
    return jsonify({"message": "User account and associated activity were removed"})


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """End the local application session."""
    user = session.get("user")
    if user:
        AuditLogger().record("user_signed_out", "success", actor_email=user.get("email"))
    session.clear()
    return jsonify({"authenticated": False})
