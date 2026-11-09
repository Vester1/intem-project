from dataclasses import replace
import json
import pytest
from fastapi.testclient import TestClient
from intem.config import Settings
from intem.graph import Application
from intem.models import Question, Plan, Draft, Claim
from intem.api import api


@pytest.fixture
def app(tmp_path):
    app = Application(Settings(db=str(tmp_path / "prototype.sqlite3")))
    path = tmp_path / "fixture.txt"
    path.write_text("ТЕСТОВЫЙ ДОКУМЕНТ. Перевод обучающихся осуществляется при наличии вакантных мест.", encoding="utf-8")
    app.retrieval.ingest(path, source="https://mephi.ru/test-fixture.pdf")
    return app


def stages(result):
    return [t["stage"] for t in result["timings"]]


def test_answer_branch_has_citations_and_no_future_agents(app):
    result = app.ask(Question(question="Как осуществляется перевод обучающихся?"))
    assert not result["refused"]
    assert "https://mephi.ru/test-fixture.pdf#page=1" in result["answer"]
    assert stages(result) == ["plan", "retrieve", "draft", "check", "answer"]
    assert not {"history", "profile", "reflections"} & result.keys()
    assert result["verification"]["semantic_check"] is False
    with app.store.connect() as db:
        tables = {r["name"] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"chunks"}


def test_no_hits_go_straight_to_refusal(app):
    result = app.ask(Question(question="Сколько стоит полёт на Марс?"))
    assert result["refused"]
    assert stages(result) == ["plan", "retrieve", "refuse"]
    assert not result["evidence"]


def test_checker_rejection_has_no_second_search(app):
    def invalid_draft(state):
        return {"draft": {"claims": [{"text": "Выдуманная норма", "quote": "Не существует", "source_id": "fake"}],
                          "missing": [], "limitations": [], "answer": ""}, "trace": state["trace"]}
    app.supervisor.draft = invalid_draft
    # Методы экземпляра регистрируются при построении графа.
    from unittest.mock import patch
    with patch("intem.agents.supervisor.SupervisorAgent.draft", side_effect=invalid_draft):
        rebuilt = Application(app.settings)
    result = rebuilt.ask(Question(question="Как осуществляется перевод обучающихся?"))
    assert result["refused"]
    assert stages(result) == ["plan", "retrieve", "draft", "check", "refuse"]
    assert result["verification"]["issues"]


class Provider:
    def __init__(self):
        self.calls = []

    def structured(self, instruction, payload, schema):
        self.calls.append(schema.__name__)
        if schema is Plan:
            return Plan(queries=["перевод"])
        assert schema is Draft  # В КП2 нет вызовов смысловой проверки или модели-судьи.
        return Draft(answer="Для перевода нужны вакантные места [1].", claims=[
            Claim(text="Для перевода нужны вакантные места.", source_id=payload["evidence"][0]["id"])])

    def chat(self, messages, tools):
        self.calls.append("tools")
        return {"role": "assistant", "tool_calls": [{"id": "one", "type": "function", "function": {
            "name": "keyword_search", "arguments": json.dumps({"query": "перевод"})}}]}


def test_llm_route_is_single_pass_and_discloses_limits(app):
    provider = Provider()
    live = Application(replace(app.settings, mode="llm"), llm=provider)
    result = live.ask(Question(question="Каковы условия перевода?"))
    assert not result["refused"]
    assert provider.calls == ["Plan", "tools", "Draft"]
    assert "Смысловая проверка не реализована" in result["answer"]
    assert "[1](https://mephi.ru/test-fixture.pdf#page=1)" in result["answer"]


def test_api_exposes_prototype_without_profile_or_website(app, monkeypatch):
    monkeypatch.setattr("intem.api.get_app", lambda: app)
    with TestClient(api) as client:
        assert client.get("/").url.path == "/docs"
        assert client.get("/health").json()["stage"] == "kp2"
        assert client.get("/graph").status_code == 200
        assert client.get("/assets/app.js").status_code == 404
        assert client.put("/users/alice/profile", json={"course": 2}).status_code == 404
        assert client.post("/ask", json={"question": ""}).status_code == 422
        response = client.post("/ask", json={"question": "Как осуществляется перевод обучающихся?"})
        assert response.status_code == 200 and not response.json()["refused"]
