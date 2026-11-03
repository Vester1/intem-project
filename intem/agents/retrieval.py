"""Владелец: Денис Иртуганов. Индексация, векторы и вызов функций."""
import hashlib
import json
import math
import re
from pathlib import Path
from pypdf import PdfReader
from ..llm import system_prompt

STOP = set("как какой какие сколько когда можно нужно что это для или при если мне меня мой быть есть чем после чтобы также его нее она они на по от до из за не ли во со об то и а в с у к о я".split())


def tokens(text):
    # Простое сопоставление русских слов по началу слова; семантические векторные представления настраиваются отдельно.
    return [w[:6] for w in re.findall(r"[а-яёa-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP]


def local_vector(text):
    vector = [0.0] * 384
    for token in tokens(text):
        value = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vector[value % len(vector)] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding dimension changed; reindex documents")
    norm = math.sqrt(sum(v*v for v in left) * sum(v*v for v in right))
    return sum(a*b for a, b in zip(left, right)) / norm if norm else 0.0


def split_chunks(text, limit=1200):
    """Сохраняет целые предложения и по возможности повторяет одно предложение в соседних фрагментах."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«0-9])", text)
    current = ""
    previous = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                yield current
                current = ""
            # Размер контекста ограничен даже для необычно длинных абзацев.
            words = sentence.split()
            for word in words:
                if len(current) + len(word) + 1 > limit and current:
                    yield current
                    current = ""
                current = (current + " " + word).strip()
            previous = ""
            continue
        if current and len(current) + len(sentence) + 1 > limit:
            yield current
            current = previous if len(previous) < limit // 3 else ""
        current = (current + " " + sentence).strip()
        previous = sentence
    if current:
        yield current


class RetrievalAgent:
    def __init__(self, store, llm, settings):
        self.store, self.llm, self.settings = store, llm, settings
        self.embedding_id = settings.embedding_model or "local-hash-384-v1"

    def vectors(self, texts):
        return self.llm.embed(texts) if self.settings.embedding_model else [local_vector(t) for t in texts]

    def ingest(self, path, source=None, metadata=None):
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            pages = [(i, page.extract_text() or "") for i, page in enumerate(PdfReader(path).pages, 1)]
        elif path.suffix.lower() in {".md", ".txt"}:
            pages = [(1, path.read_text(encoding="utf-8"))]
        else:
            raise ValueError("Supported documents: PDF, TXT, MD")
        if not any(len(text.strip()) > 40 for _, text in pages):
            raise ValueError(f"No text layer in {path.name}; OCR required before indexing")
        source = source or str(path.resolve())
        metadata = {**(metadata or {}), "embedding_model": self.embedding_id}
        records = []
        for page, text in pages:
            text = re.sub(r"\s+", " ", text).strip()
            for start, chunk in enumerate(split_chunks(text)):
                if len(chunk) < 40:
                    continue
                identifier = hashlib.sha256(f"{source}:{page}:{start}:{chunk}".encode()).hexdigest()[:20]
                records.append((identifier, source, page, chunk))
        if not records:
            raise ValueError("No indexable text")
        vectors = self.vectors([r[3] for r in records])
        if len(vectors) != len(records):
            raise ValueError("Embedding response count mismatch")
        with self.store.connect() as db:
            db.execute("DELETE FROM chunks WHERE source=?", (source,))
            db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                           [(*r, json.dumps(v), json.dumps(metadata, ensure_ascii=False)) for r, v in zip(records, vectors)])
        return len(records)

    def search(self, query, method="hybrid", limit=6):
        if method not in {"hybrid", "keyword", "vector"}:
            raise ValueError("Unknown search method")
        query_tokens = set(tokens(query))
        vector = self.vectors([query])[0] if method != "keyword" else None
        results = []
        for row in self.store.chunks():
            if row["metadata"].get("embedding_model") != self.embedding_id:
                raise ValueError("Embedding model changed; reindex the corpus")
            lexical = len(query_tokens & set(tokens(row["text"]))) / max(1, len(query_tokens))
            semantic = cosine(vector, row["vector"]) if vector else 0.0
            score = lexical if method == "keyword" else semantic if method == "vector" else .55*lexical + .45*semantic
            # Коллизии локальных хешей не должны считаться признаком релевантности.
            if score < .12 or (not self.settings.embedding_model and lexical == 0):
                continue
            results.append({k: v for k, v in row.items() if k != "vector"} | {"score": round(score, 4)})
        return sorted(results, key=lambda r: (-r["score"], r["id"]))[:limit]

    def run(self, state):
        queries = state.get("verification", {}).get("retry_queries") or state["queries"]
        found, events = {}, []
        if self.settings.mode == "demo":
            for query in queries:
                for row in self.search(query, limit=10):
                    found[row["id"]] = row
            events.append({"agent": "retrieval", "event": "search", "tool": "hybrid_search", "queries": queries})
        else:
            tools = [{"type": "function", "function": {"name": name, "description": description,
                     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}}
                     for name, description in [("hybrid_search", "Гибридный поиск норм"), ("keyword_search", "Точное совпадение терминов и номеров"), ("vector_search", "Поиск по сходству векторов")]]
            messages = [{"role": "system", "content": system_prompt() + f"\nТы Retrieval Agent. Найди нормы по всем подзадачам. Объединяй независимые запросы в один раунд вызовов инструментов. При достаточных источниках сразу завершай поиск. Меняй запрос только при недостатке данных. Максимум {1} раунда инструментов."},
                        {"role": "user", "content": json.dumps({"question": state["question"], "queries": queries, "prototype": "kp2"}, ensure_ascii=False)}]
            budget = 12
            for _ in range(1):
                message = self.llm.chat(messages, tools)
                calls = message.get("tool_calls", [])
                if not calls:
                    break
                if len(calls) > budget:
                    raise ValueError("Retrieval tool budget exceeded")
                budget -= len(calls)
                messages.append({k: message[k] for k in ("role", "content", "tool_calls") if k in message})
                for call in calls:
                    name = call["function"]["name"]
                    methods = {"hybrid_search": "hybrid", "keyword_search": "keyword", "vector_search": "vector"}
                    if name not in methods:
                        raise ValueError("Unknown tool requested")
                    arguments = json.loads(call["function"]["arguments"])
                    query = arguments["query"]
                    if not isinstance(query, str) or not 1 <= len(query) <= 6000:
                        raise ValueError("Invalid retrieval query")
                    rows = self.search(query, methods[name], limit=8)
                    for row in rows:
                        found[row["id"]] = row
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(rows, ensure_ascii=False)})
                    events.append({"agent": "retrieval", "event": "tool_call", "tool": name, "query": query, "hits": len(rows)})
        evidence = sorted(found.values(), key=lambda r: -r["score"])[:24]
        return {"evidence": evidence, "trace": state["trace"] + events + [{"agent": "retrieval", "event": "evidence_ready", "count": len(evidence)}]}
