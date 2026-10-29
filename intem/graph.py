"""Сергей Королев. Базовый граф КП2 с двумя конечными маршрутами."""
from langgraph.graph import StateGraph, START, END
from .agents.supervisor import SupervisorAgent
from .agents.retrieval import RetrievalAgent
from .agents.fact_checker import FactCheckerAgent
from .config import Settings
from .llm import LLM
from .models import State, Question
from .storage import Store
from .runtime import request_budget, CURRENT_BUDGET
import time


class Application:
    def __init__(self, settings=None, llm=None):
        self.settings = settings or Settings.from_env()
        self.store = Store(self.settings.db)
        self.llm = llm or LLM(self.settings)
        self.retrieval = RetrievalAgent(self.store, self.llm, self.settings)
        self.supervisor = SupervisorAgent(self.llm, self.settings)
        self.checker = FactCheckerAgent(self.llm, self.settings)
        builder = StateGraph(State)
        for name, node in [("plan", self.supervisor.plan), ("retrieve", self.retrieval.run),
                           ("draft", self.supervisor.draft), ("check", self.checker.run),
                           ("answer", self.supervisor.finish), ("refuse", self.supervisor.finish)]:
            builder.add_node(name, self.observed(name, node))
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_conditional_edges("retrieve", self.after_retrieval,
                                      {"draft": "draft", "refuse": "refuse"})
        builder.add_edge("draft", "check")
        builder.add_conditional_edges("check", self.after_check,
                                      {"answer": "answer", "refuse": "refuse"})
        builder.add_edge("answer", END)
        builder.add_edge("refuse", END)
        self.graph = builder.compile()

    @staticmethod
    def after_retrieval(state):
        return "draft" if state["evidence"] else "refuse"

    @staticmethod
    def after_check(state):
        return "answer" if state["verification"]["supported"] else "refuse"

    @staticmethod
    def observed(name, node):
        def run(state):
            budget = CURRENT_BUDGET.get()
            if budget:
                budget.remaining()
                if budget.notify:
                    budget.notify({"event": "stage_start", "stage": name})
            started = time.monotonic()
            result = node(state)
            if budget:
                budget.remaining()
            result["timings"] = state.get("timings", []) + [
                {"stage": name, "duration_seconds": round(time.monotonic()-started, 3)}]
            return result
        return run

    def ask(self, question: Question, on_progress=None):
        with request_budget(self.settings.request_timeout, self.settings.max_llm_calls, on_progress):
            return self.graph.invoke({**question.model_dump(), "trace": [],
                "verification": {"supported": False, "issues": ["Поиск не нашёл подходящих фрагментов."],
                                 "kind": "citation_format_only", "semantic_check": False}},
                config={"recursion_limit": 12})
