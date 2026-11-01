import re
from ..models import Draft


class FactCheckerAgent:
    def __init__(self, llm, settings):
        self.settings = settings

    def run(self, state):
        draft = Draft.model_validate(state["draft"])
        evidence = {row["id"]: row for row in state["evidence"]}
        issues = []
        if not draft.claims:
            issues.append("Нет фактов с источниками.")
        if draft.missing:
            issues.append("Не найдены сведения: " + "; ".join(draft.missing))
        for claim in draft.claims:
            source = evidence.get(claim.source_id)
            if not source or not claim.quote or claim.quote not in source["text"]:
                issues.append("Источник не найден или цитата отсутствует в его тексте.")
            if self.settings.mode == "demo" and claim.text != claim.quote:
                issues.append("В demo допустимы только дословные выдержки.")
        if self.settings.mode == "llm":
            markers = re.findall(r"\[([^\]]*)\]", draft.answer)
            expected = {str(i) for i in range(1, len(draft.claims) + 1)}
            if not draft.answer.strip() or set(markers) != expected:
                issues.append("Отсутствует ответ или нарушена нумерация ссылок.")
            if re.search(r"https?://", draft.answer, re.I) or any(
                    re.search(r"https?://|\[[^\]]*\]", item, re.I) for item in draft.limitations):
                issues.append("Ссылки формирует программа из индекса.")
        verification = {"supported": not issues, "issues": issues,
                        "kind": "citation_format_only", "semantic_check": False}
        return {"verification": verification,
                "trace": state["trace"] + [{"agent": "fact_checker",
                    "event": "accepted" if not issues else "rejected", "issues": issues}]}
