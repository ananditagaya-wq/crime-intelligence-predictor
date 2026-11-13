
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import re

from auth.database import create_user, get_user_by_email
from auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    verify_access_token
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

bearer_scheme = HTTPBearer(auto_error=False)


# Request models
class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


# Email validation helper
def normalize_email(email: str):
    email = email.strip().lower()

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(
            status_code=422,
            detail="Please enter a valid email address."
        )

    return email


# Signup endpoint
@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(data: SignupRequest):

    name = data.name.strip()
    email = normalize_email(data.email)

    if len(name) < 2:
        raise HTTPException(status_code=422, detail="Name is too short.")

    if get_user_by_email(email):
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    hashed_password = hash_password(data.password)

    user_id = create_user(name, email, hashed_password)

    if user_id is None:
        raise HTTPException(status_code=409, detail="Email already registered.")

    return {
        "message": "Account created successfully.",
        "user_id": user_id,
        "name": name,
        "email": email
    }


# Login endpoint
@router.post("/login")
def login(data: LoginRequest):

    email = normalize_email(data.email)
    user = get_user_by_email(email)

    if not user or not verify_password(
        data.password,
        user["hashed_password"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password."
        )

    token = create_access_token(user["id"], user["email"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"]
        }
    }


# Authentication dependency for protected endpoints
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    payload = verify_access_token(credentials.credentials)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token."
        )

    user = get_user_by_email(payload.get("email", ""))

    if not user or str(user["id"]) != payload.get("sub"):
        raise HTTPException(
            status_code=401,
            detail="User account not found."
        )

    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"]
    }


# Current logged-in user endpoint
@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return current_user
