"""FastAPI server for the farm LLM assistant."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from ai_services.computer_vision import analyze_cows
from ai_services.llm import GeminiClientError, generate_llm_answer
from schemas import AnalyzeResponse, AskQuestionParams, parse_add_info

app = FastAPI(title="Farm LLM Assistant")
_ASK_REQUEST_LOCK = Lock()
_ANALYZE_REQUEST_LOCK = Lock()


@contextmanager
def single_flight(lock: Lock) -> None:
    """Allow only one request at a time for guarded endpoints."""

    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Service is busy processing another request. Try again later.",
        )
    try:
        yield
    finally:
        lock.release()


@app.get("/ask-question")
def ask_question(params: AskQuestionParams = Depends()) -> dict[str, str]:
    """Return an LLM answer to the farmer's question."""

    with single_flight(_ASK_REQUEST_LOCK):
        try:
            answer = generate_llm_answer(params.history, params.user_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GeminiClientError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"llm_answer": answer}


@app.get("/analyze", response_model=AnalyzeResponse)
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request) -> AnalyzeResponse:
    """Analyze a cow image with optional extra metadata."""

    with single_flight(_ANALYZE_REQUEST_LOCK):
        try:
            form = await request.form(max_part_size=5 * 1024 * 1024)
        except HTTPException as exc:
            raise exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=400, detail="Invalid multipart form data"
            ) from exc

        image = form.get("image")
        if image is None:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "missing",
                        "loc": ["body", "image"],
                        "msg": "Field required",
                        "input": None,
                    }
                ],
            )
        if not isinstance(image, UploadFile):
            raise HTTPException(
                status_code=400, detail="image must be sent as a file upload"
            )

        add_info_raw = form.get("add_info")
        try:
            add_info_payload = parse_add_info(add_info_raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            pil_image = Image.open(image.file)
            pil_image.load()
        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=400, detail="image must be a valid image file"
            ) from exc
        finally:
            image.file.close()

        return analyze_cows(pil_image, add_info_payload)
