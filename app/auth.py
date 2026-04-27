from __future__ import annotations

from typing import Annotated

from fastapi import Header, HTTPException

from .settings import get_settings


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = get_settings().api_key
    supplied = x_api_key
    if supplied is None and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            supplied = token.strip()

    if not expected or supplied != expected:
        raise HTTPException(status_code=401, detail="Valid API key required.")
