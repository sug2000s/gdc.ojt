from fastapi import HTTPException


def error_detail(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def raise_api_error(status_code: int, code: str, message: str) -> None:
    raise HTTPException(status_code=status_code, detail=error_detail(code, message))