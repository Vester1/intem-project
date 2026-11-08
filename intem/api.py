"""КП2: синхронный API и Swagger без веб-клиента."""
from functools import lru_cache
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from .graph import Application
from .models import Question
from .runtime import RequestDeadlineExceeded, ProviderTimedOut, CallBudgetExceeded

api = FastAPI(title="Intem · Прототип КП2", version="0.0.2")


@lru_cache
def get_app():
    return Application()


@api.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@api.get("/health")
def health():
    app = get_app()
    with app.store.connect() as db:
        row = db.execute("SELECT COUNT(*) AS chunks, COUNT(DISTINCT source) AS sources FROM chunks").fetchone()
    return {"status": "ok", "stage": "kp2", "mode": app.settings.mode,
            "model": app.settings.model if app.settings.mode == "llm" else "demo", **dict(row)}


@api.get("/graph")
def graph():
    return {"mermaid": get_app().graph.get_graph().draw_mermaid()}


@api.post("/ask")
def ask(question: Question):
    try:
        return get_app().ask(question)
    except (RequestDeadlineExceeded, ProviderTimedOut) as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except CallBudgetExceeded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=502, detail="Ошибка провайдера или формата ответа") from exc
