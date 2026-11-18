from fastapi import HTTPException
from fastapi.responses import JSONResponse


class OpenAIAPIError(HTTPException):
    def __init__(
        self,
        *,
        status: int,
        message: str,
        type: str,
        code: int | str,
    ) -> None:
        super().__init__(status_code=status, detail=message)
        self.message = message
        self.type = type
        self.code = code


def openai_error(
    status: int,
    message: str,
    type: str,
    code: int | str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": type,
                "code": code,
            }
        },
    )
