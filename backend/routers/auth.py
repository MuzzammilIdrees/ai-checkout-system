"""
routers/auth.py — Authentication endpoints for the AI Checkout System.

Implements a two-step MFA login flow:
1. POST /auth/login   — validate credentials, send OTP to email
2. POST /auth/verify   — verify OTP, return JWT access token
3. POST /auth/validate — validate an existing JWT token
4. POST /auth/logout   — invalidate active OTP sessions

Security features:
- SHA-256 + salt password hashing (HMAC)
- Account lockout after 5 failed attempts (15-minute lockout)
- Rate limiting on login/verify endpoints
- OTP brute-force protection (max 5 attempts per OTP session)
- Constant-time password comparison
- JWT tokens with configurable expiry
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from database.db import AdminUser, OTPSession
from models.schemas import (
    ErrorResponse,
    LoginRequest,
    LoginResponse,
    OTPVerifyRequest,
    OTPVerifyResponse,
    TokenValidationResponse,
)
from utils.security import (
    MAX_LOGIN_ATTEMPTS,
    MAX_OTP_ATTEMPTS,
    OTP_EXPIRY_MINUTES,
    create_jwt_token,
    generate_otp,
    generate_session_token,
    get_lockout_time,
    get_otp_expiry,
    hash_otp,
    is_account_locked,
    send_otp_email,
    verify_jwt_token,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# Rate limiter for auth endpoints (uses the app-level limiter)
limiter = Limiter(key_func=get_remote_address)


def _mask_email(email: str) -> str:
    """Mask an email address for display (e.g. m***@gmail.com)."""
    if "@" not in email:
        return "***"
    local, domain = email.rsplit("@", 1)
    if len(local) <= 2:
        masked = local[0] + "***"
    else:
        masked = local[0] + "***" + local[-1]
    return f"{masked}@{domain}"


@router.post(
    "/login",
    response_model=LoginResponse,
    responses={
        401: {"model": ErrorResponse},
        423: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    summary="Admin login (step 1 of MFA)",
    description=(
        "Validate admin credentials and send a 6-digit OTP to the registered email. "
        "Returns a session_token to use in the /verify step."
    ),
)
@limiter.limit("5/minute")
async def login(request: Request, body: LoginRequest) -> LoginResponse:
    """
    Step 1 of MFA login: validate username/password and send OTP.

    Args:
        request: FastAPI request (used by rate limiter).
        body: LoginRequest with username and password.

    Returns:
        LoginResponse with session_token and masked email hint.

    Raises:
        HTTPException 401: Invalid credentials.
        HTTPException 423: Account locked due to too many failed attempts.
    """
    from main import get_db_session

    db = get_db_session()

    try:
        # Find user
        user = db.query(AdminUser).filter(AdminUser.username == body.username).first()

        if user is None:
            logger.warning(f"⚠️ Login attempt for unknown user: '{body.username}'")
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password.",
            )

        # Check account lockout
        if is_account_locked(user.locked_until):
            logger.warning(f"🔒 Locked account login attempt: '{body.username}'")
            raise HTTPException(
                status_code=423,
                detail="Account is locked due to too many failed attempts. Try again later.",
            )

        # Check if account is active
        if not user.is_active:
            raise HTTPException(
                status_code=401,
                detail="Account is deactivated. Contact system administrator.",
            )

        # Verify password
        if not verify_password(body.password, user.salt, user.password_hash):
            # Increment failed attempts
            user.failed_attempts += 1
            if user.failed_attempts >= MAX_LOGIN_ATTEMPTS:
                user.locked_until = get_lockout_time()
                logger.warning(
                    f"🔒 Account locked: '{body.username}' after {user.failed_attempts} failed attempts"
                )
            db.commit()

            remaining = MAX_LOGIN_ATTEMPTS - user.failed_attempts
            logger.warning(
                f"⚠️ Failed login for '{body.username}' "
                f"(attempt {user.failed_attempts}/{MAX_LOGIN_ATTEMPTS})"
            )
            raise HTTPException(
                status_code=401,
                detail=f"Invalid username or password. {max(0, remaining)} attempts remaining.",
            )

        # Reset failed attempts on successful password verification
        user.failed_attempts = 0
        user.locked_until = None

        # Generate OTP
        otp_code = generate_otp()
        session_token = generate_session_token()

        # Invalidate any existing OTP sessions for this user
        db.query(OTPSession).filter(
            OTPSession.user_id == user.id,
            OTPSession.is_used == False,
        ).update({"is_used": True})

        # Create new OTP session
        otp_session = OTPSession(
            user_id=user.id,
            otp_hash=hash_otp(otp_code),
            expires_at=get_otp_expiry(),
            session_token=session_token,
        )
        db.add(otp_session)
        db.commit()

        # Send OTP via email (falls back to console if SMTP not configured)
        send_otp_email(user.email, otp_code, user.username)

        logger.info(f"✅ OTP sent for user '{body.username}' to {_mask_email(user.email)}")

        return LoginResponse(
            message="OTP sent to registered email",
            session_token=session_token,
            email_hint=_mask_email(user.email),
            otp_expiry_minutes=OTP_EXPIRY_MINUTES,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during login.")
    finally:
        db.close()


@router.post(
    "/verify",
    response_model=OTPVerifyResponse,
    responses={
        401: {"model": ErrorResponse},
        410: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    summary="Verify OTP (step 2 of MFA)",
    description=(
        "Submit the 6-digit OTP received via email along with the session_token "
        "from the login step. Returns a JWT access token on success."
    ),
)
@limiter.limit("10/minute")
async def verify_otp(request: Request, body: OTPVerifyRequest) -> OTPVerifyResponse:
    """
    Step 2 of MFA login: verify OTP and issue JWT.

    Args:
        request: FastAPI request (used by rate limiter).
        body: OTPVerifyRequest with session_token and otp_code.

    Returns:
        OTPVerifyResponse with JWT access token.

    Raises:
        HTTPException 401: Invalid OTP or session.
        HTTPException 410: OTP expired.
    """
    from main import get_db_session

    db = get_db_session()

    try:
        # Find OTP session
        otp_session = (
            db.query(OTPSession)
            .filter(OTPSession.session_token == body.session_token)
            .first()
        )

        if otp_session is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired session. Please login again.",
            )

        # Check if already used
        if otp_session.is_used:
            raise HTTPException(
                status_code=401,
                detail="This OTP session has already been used. Please login again.",
            )

        # Check expiry
        now = datetime.now(timezone.utc)
        expires_at = otp_session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if now > expires_at:
            otp_session.is_used = True
            db.commit()
            raise HTTPException(
                status_code=410,
                detail="OTP has expired. Please login again to receive a new code.",
            )

        # Check brute-force protection
        if otp_session.attempts >= MAX_OTP_ATTEMPTS:
            otp_session.is_used = True
            db.commit()
            raise HTTPException(
                status_code=401,
                detail="Too many verification attempts. Please login again.",
            )

        # Verify OTP
        otp_session.attempts += 1

        if hash_otp(body.otp_code) != otp_session.otp_hash:
            remaining = MAX_OTP_ATTEMPTS - otp_session.attempts
            db.commit()
            raise HTTPException(
                status_code=401,
                detail=f"Invalid verification code. {remaining} attempts remaining.",
            )

        # OTP is valid — mark as used
        otp_session.is_used = True

        # Get user and update last_login
        user = db.query(AdminUser).filter(AdminUser.id == otp_session.user_id).first()
        if user:
            user.last_login = now

        db.commit()

        # Generate JWT token
        from utils.security import JWT_EXPIRY_MINUTES

        token = create_jwt_token(user.id, user.username)

        logger.info(f"✅ MFA login complete for '{user.username}'")

        return OTPVerifyResponse(
            message="Login successful",
            access_token=token,
            token_type="bearer",
            expires_in_minutes=JWT_EXPIRY_MINUTES,
            username=user.username,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ OTP verification error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during verification.")
    finally:
        db.close()


@router.post(
    "/validate",
    response_model=TokenValidationResponse,
    summary="Validate JWT token",
    description="Check if a JWT token is still valid. Used by the frontend to verify session state.",
)
async def validate_token(request: Request) -> TokenValidationResponse:
    """
    Validate a JWT access token from the Authorization header.

    Args:
        request: FastAPI request with Authorization header.

    Returns:
        TokenValidationResponse with validity and user info.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return TokenValidationResponse(valid=False)

    token = auth_header[7:]  # Strip "Bearer "
    payload = verify_jwt_token(token)

    if payload is None:
        return TokenValidationResponse(valid=False)

    return TokenValidationResponse(
        valid=True,
        username=payload.get("username", ""),
        user_id=int(payload.get("sub", 0)),
    )
