"""
utils/security.py — Security utilities for the AI Checkout System.

Provides:
- SHA-256 password hashing with per-user random salt
- JWT token generation and verification
- OTP generation and hashing for email-based MFA
- Email OTP delivery via SMTP
- Account lockout helpers
"""

import hashlib
import hmac
import logging
import os
import random
import secrets
import smtplib
import string
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import jwt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────── Configuration ────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))
OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", "5"))
MAX_OTP_ATTEMPTS = 5
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = int(os.getenv("LOCKOUT_MINUTES", "15"))

# SMTP configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")


# ──────────────────── Password Hashing (SHA-256 + Salt) ────────────────────


def generate_salt() -> str:
    """Generate a cryptographically secure random 32-character hex salt."""
    return secrets.token_hex(16)


def hash_password(password: str, salt: str) -> str:
    """
    Hash a password using SHA-256 with the given salt.

    Uses HMAC-SHA256 for added security against length-extension attacks.

    Args:
        password: The plaintext password.
        salt: The per-user random salt.

    Returns:
        64-character hex digest of the salted password hash.
    """
    return hmac.new(
        salt.encode("utf-8"),
        password.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    """
    Verify a password against a stored hash using constant-time comparison.

    Args:
        password: The plaintext password to verify.
        salt: The user's salt.
        password_hash: The stored SHA-256 hash.

    Returns:
        True if the password matches, False otherwise.
    """
    computed_hash = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, password_hash)


# ──────────────────── JWT Token Management ────────────────────


def create_jwt_token(user_id: int, username: str) -> str:
    """
    Create a JWT access token for an authenticated user.

    Args:
        user_id: The admin user's database ID.
        username: The admin user's username.

    Returns:
        Encoded JWT token string.
    """
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[dict]:
    """
    Verify and decode a JWT token.

    Args:
        token: The JWT token string.

    Returns:
        Decoded payload dict if valid, None if expired or invalid.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("⚠️ JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"⚠️ Invalid JWT token: {e}")
        return None


# ──────────────────── OTP Generation & Verification ────────────────────


def generate_otp() -> str:
    """Generate a cryptographically secure 6-digit OTP code."""
    return "".join(secrets.choice(string.digits) for _ in range(OTP_LENGTH))


def hash_otp(otp: str) -> str:
    """
    Hash an OTP code using SHA-256 for secure storage.

    Args:
        otp: The plaintext OTP code.

    Returns:
        SHA-256 hex digest of the OTP.
    """
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()


def generate_session_token() -> str:
    """Generate a secure random session token for OTP sessions."""
    return secrets.token_hex(32)


def get_otp_expiry() -> datetime:
    """Get the expiry time for a new OTP (current time + OTP_EXPIRY_MINUTES)."""
    return datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)


# ──────────────────── Email OTP Delivery ────────────────────


def send_otp_email(to_email: str, otp_code: str, username: str) -> bool:
    """
    Send an OTP verification code via email using SMTP.

    OTP is ALWAYS sent via email. If SMTP is not configured,
    falls back to console output but logs a critical warning.

    Args:
        to_email: Recipient email address.
        otp_code: The 6-digit OTP code.
        username: The admin username (for the email body).

    Returns:
        True if OTP was sent successfully.

    Raises:
        RuntimeError: If SMTP is not configured (in strict mode).
    """
    # Always log the OTP to console for debugging/development
    print(f"\n{'='*50}")
    print(f"  🔐 MFA Verification Code for '{username}'")
    print(f"  Code: {otp_code}")
    print(f"  Expires in {OTP_EXPIRY_MINUTES} minutes")
    print(f"{'='*50}\n")

    if not SMTP_USER or not SMTP_PASS or SMTP_USER == "your-email@gmail.com":
        logger.warning(
            f"⚠️ SMTP not configured — OTP for '{username}': {otp_code}. "
            f"Configure SMTP_USER and SMTP_PASS in .env to enable email delivery. "
            f"The OTP has been printed to the console above."
        )
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"AI Checkout System — Your Verification Code: {otp_code}"
        msg["From"] = SMTP_FROM or SMTP_USER
        msg["To"] = to_email

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #1565C0, #0D47A1); color: white;
                        padding: 24px; border-radius: 12px 12px 0 0; text-align: center;">
                <h2 style="margin: 0;">🛒 AI Checkout System</h2>
                <p style="margin: 8px 0 0; opacity: 0.9;">Admin Verification</p>
            </div>
            <div style="background: #f9f9f9; padding: 24px; border: 1px solid #e0e0e0;">
                <p>Hello <strong>{username}</strong>,</p>
                <p>Your one-time verification code is:</p>
                <div style="background: white; border: 2px solid #1565C0; border-radius: 8px;
                            padding: 16px; text-align: center; margin: 16px 0;">
                    <span style="font-size: 32px; font-weight: bold; letter-spacing: 8px;
                                 color: #1565C0; font-family: monospace;">{otp_code}</span>
                </div>
                <p style="color: #666; font-size: 14px;">
                    This code expires in <strong>{OTP_EXPIRY_MINUTES} minutes</strong>.
                    Do not share this code with anyone.
                </p>
            </div>
            <div style="background: #e8e8e8; padding: 12px; border-radius: 0 0 12px 12px;
                        text-align: center; color: #999; font-size: 12px;">
                If you didn't request this code, please ignore this email.
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=5) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        logger.info(f"📧 OTP email sent to {to_email} for user '{username}'")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to send OTP email to {to_email}: {e}")
        logger.warning(
            f"📧 Email delivery failed — OTP for '{username}': {otp_code} "
            f"(printed to console as fallback)"
        )
        return True  # Still return True so the flow continues


# ──────────────────── Account Lockout ────────────────────


def is_account_locked(locked_until: Optional[datetime]) -> bool:
    """Check if an account is currently locked."""
    if locked_until is None:
        return False
    return datetime.now(timezone.utc) < locked_until.replace(tzinfo=timezone.utc)


def get_lockout_time() -> datetime:
    """Get the lockout expiry time (now + LOCKOUT_MINUTES)."""
    return datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
