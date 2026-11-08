"""Лимиты каждого запроса изолированы от параллельных запросов к API."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time


class RequestDeadlineExceeded(TimeoutError):
    pass


class ProviderTimedOut(TimeoutError):
    pass


class CallBudgetExceeded(RuntimeError):
    pass


@dataclass
class RequestBudget:
    seconds: float
    max_calls: int
    notify: object = None
    started: float = field(default_factory=time.monotonic)
    calls: int = 0

    def remaining(self):
        remaining = self.seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise RequestDeadlineExceeded("Истёк общий лимит времени запроса. Подтверждённый ответ не получен.")
        return remaining

    def begin_call(self):
        remaining = self.remaining()
        if self.calls >= self.max_calls:
            raise CallBudgetExceeded("Достигнут лимит обращений к модели. Подтверждённый ответ не получен.")
        self.calls += 1
        if self.notify:
            self.notify({"event": "llm_call", "call": self.calls, "elapsed": time.monotonic() - self.started})
        return remaining


CURRENT_BUDGET = ContextVar("intem_request_budget", default=None)


@contextmanager
def request_budget(seconds, max_calls, notify=None):
    budget = RequestBudget(seconds, max_calls, notify)
    token = CURRENT_BUDGET.set(budget)
    try:
        yield budget
    finally:
        CURRENT_BUDGET.reset(token)
