"""FastAPI server for the farm LLM assistant."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from ai_services.llm import GeminiClientError, generate_llm_answer
from schemas import AskQuestionParams

app = FastAPI(title="Farm LLM Assistant")


@app.get("/ask-question")
def ask_question(params: AskQuestionParams = Depends()) -> dict[str, str]:
    """Return an LLM answer to the farmer's question."""

    try:
        answer = generate_llm_answer(params.history, params.user_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GeminiClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"llm_answer": answer}
