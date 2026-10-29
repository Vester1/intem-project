"""Владелец: Сергей Королев. План, синтез и управление графом."""
import re
from ..models import Plan, Draft, Claim
from .retrieval import tokens
from ..citations import source_link, source_location


class SupervisorAgent:
    def __init__(self, llm, settings):
        self.llm, self.settings = llm, settings

    def plan(self, state):
        if self.settings.mode == "llm":
            result = self.llm.structured("Ты Supervisor прототипа. Разложи вопрос по нормативной базе НИЯУ МИФИ на 1–3 поисковых запроса. Нормализуй разговорные слова: хвосты — академическая задолженность, вылететь — отчисление. Факты не добавляй.",
                                         {"question": state["question"]}, Plan)
            queries = result.queries
        else:
            queries = [s.strip(" ?") for s in re.split(r"\?|;| и (?=как|можно|когда|сколько)", state["question"]) if len(s.strip()) > 2][:5]
        return {"queries": queries, "trace": state["trace"] + [{"agent": "supervisor", "event": "plan", "queries": queries}]}

    def draft(self, state):
        # Короткие ссылки внутри черновика избавляют небольшую модель от копирования непрозрачных хеш-идентификаторов.
        source_refs = {f"E{i}": item for i, item in enumerate(state["evidence"], 1)}
        if self.settings.mode == "llm":
            result = self.llm.structured("""Составь краткий ответ по evidence, без общих знаний.
В answer ставь ссылки [1], [2] по порядку claims. Каждый нормативный факт внеси
в claims: text — факт, source_id — E1/E2 из evidence. quote оставь пустым,
его заполнит программа. Не вставляй URL. Учитывай условия и даты источника.
Если данных недостаточно, верни пустые claims и missing с причиной.
Это однопроходный прототип: дополнительных поисков и исправлений не будет.""",
                                         {"question": state["question"], "queries": state["queries"],
                                          "evidence": [{**item, "id": ref} for ref, item in source_refs.items()]}, Draft)
        else:
            claims, missing = [], []
            for query in state["queries"]:
                query_tokens = set(tokens(query))
                candidates = []
                for evidence in state["evidence"]:
                    for sentence in re.split(r"(?<=[.!?])\s+", evidence["text"]):
                        overlap = len(query_tokens & set(tokens(sentence)))
                        if overlap and len(sentence) >= 30:
                            candidates.append((overlap, sentence, evidence["id"]))
                candidates.sort(key=lambda c: -c[0])
                matched_tokens = set()
                for _, sentence, _ in candidates[:2]:
                    matched_tokens |= query_tokens & set(tokens(sentence))
                # Совпадение одного слова не должно превращать посторонний фрагмент в ответ.
                # Это осторожная лексическая эвристика, а не смысловая проверка.
                if not candidates or len(matched_tokens) / max(1, len(query_tokens)) < .5:
                    missing.append(query)
                    continue
                for _, sentence, identifier in candidates[:2]:
                    claim = Claim(text=sentence, source_id=identifier, quote=sentence)
                    if claim not in claims:
                        claims.append(claim)
            result = Draft(claims=claims[:15], missing=missing)
        if self.settings.mode == "llm":
            # Текст источника берётся из индекса и никогда не восстанавливается моделью.
            evidence_by_id = {item["id"]: item for item in state["evidence"]}
            for claim in result.claims:
                source = source_refs.get(claim.source_id) or evidence_by_id.get(claim.source_id)
                if source:
                    claim.source_id = source["id"]
                    claim.quote = source["text"]
            if result.answer.strip() and result.claims:
                if not re.search(r"\[[^\]]*\]", result.answer):
                    result.answer += " " + " ".join(f"[{i}]" for i in range(1, len(result.claims) + 1))
                # Подвопросы без ответа явно перечисляются, а не скрываются.
                # КП2: техническая проверка ссылок без независимого LLM-судьи.
                result.limitations = list(dict.fromkeys(result.limitations + result.missing))
                result.missing = []
        return {"draft": result.model_dump(), "trace": state["trace"] + [{"agent": "supervisor", "event": "draft", "claims": len(result.claims)}]}

    def finish(self, state):
        accepted = state["verification"]["supported"]
        if accepted:
            evidence = {e["id"]: e for e in state["evidence"]}
            lines = []
            sources = []
            for number, c in enumerate(state["draft"]["claims"], 1):
                source = evidence[c["source_id"]]
                title = source["metadata"].get("title", source["source"])
                location, link = source_location(source), source_link(source)
                lines.append(f'{c["text"]}\nИсточник: {title}, {location}. {link}')
                sources.append((title, location, link))
            if self.settings.mode == "llm":
                def cite(match):
                    number = int(match.group(1))
                    return f'[{number}]({sources[number-1][2]})'
                answer = re.sub(r"\[(\d+)\]", cite, state["draft"]["answer"].strip())
                if state["draft"].get("limitations"):
                    answer += "\n\nОграничения ответа:\n" + "\n".join(f"- {item}" for item in state["draft"]["limitations"])
                answer += "\n\nИсточники:\n" + "\n".join(
                    f"{i}. {title}, {location}." for i, (title, location, _) in enumerate(sources, 1))
            else:
                answer = "\n\n".join(lines)
        else:
            answer = "Прототип не смог подготовить ответ по загруженным документам. " + " ".join(state["verification"]["issues"])
        if self.settings.mode == "demo":
            answer = "[Демонстрационный режим: поиск и дословные выдержки, без LLM-анализа применимости.]\n\n" + answer
        if accepted and self.settings.mode == "llm":
            answer += "\n\n[Прототип КП2: выполнена техническая проверка цитат. Смысловая проверка не реализована.]"
        status = "refused" if not accepted else "partial" if state["draft"].get("limitations") else "complete"
        return {"answer": answer, "refused": not accepted, "answer_status": status,
                "trace": state["trace"] + [{"agent": "supervisor", "event": "refusal" if not accepted else "answer"}]}
