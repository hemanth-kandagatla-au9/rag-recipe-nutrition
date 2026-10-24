"""Test doubles so the pipeline plumbing can be tested without downloading models or calling Groq.
They test OUR code (filtering, merging, refusal, citations), not retrieval quality."""
import hashlib
import re
import sys
import types

import numpy as np


class FakeEmbedder:
    dim = 128

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class FakeReranker:
    """Score = word overlap with the query (positive when relevant, -10 when nothing overlaps)."""

    def rerank(self, query, candidates, top_k=5):
        q = set(re.findall(r"[a-z0-9]+", query.lower())) - {"the", "a", "of", "in", "is", "how", "much", "what"}
        out = []
        for c in candidates:
            w = set(re.findall(r"[a-z0-9]+", c["text"].lower()))
            ov = len(q & w)
            out.append({**c, "rerank_score": float(ov) if ov else -10.0})
        out.sort(key=lambda c: (-c["rerank_score"], c["retrieval_rank"]))
        for r, c in enumerate(out, 1):
            c["rerank_rank"] = r
        return out[:top_k]


class FakeLLM:
    """Echoes a cited answer from block [1] so we can test citation extraction; counts calls."""

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):  # used directly by the router
        self.calls += 1
        return types.SimpleNamespace(content="Answer from the context [1].")

    def __call__(self, prompt_value):  # lets `prompt | FakeLLM()` work: LCEL wraps callables in RunnableLambda
        return self.invoke(prompt_value).content


def install_fake_faiss():
    """Minimal numpy stand-in for the three faiss functions Retriever/build_index use."""
    mod = types.ModuleType("faiss")

    class IndexFlatIP:
        def __init__(self, d):
            self.d, self.v = d, np.zeros((0, d), dtype="float32")

        @property
        def ntotal(self):
            return len(self.v)

        def add(self, x):
            self.v = np.vstack([self.v, x])

        def search(self, q, k):
            s = q @ self.v.T
            ids = np.argsort(-s, axis=1, kind="stable")[:, :k]
            return np.take_along_axis(s, ids, axis=1), ids

    def write_index(idx, path):
        np.save(path + ".npy", idx.v)

    def read_index(path):
        i = IndexFlatIP(1)
        i.v = np.load(path + ".npy")
        i.d = i.v.shape[1]
        return i

    mod.IndexFlatIP, mod.write_index, mod.read_index = IndexFlatIP, write_index, read_index
    sys.modules["faiss"] = mod


def install_fake_langchain_if_missing():
    """Tiny stand-ins for the LCEL pieces chain.py uses, ONLY when langchain-core is not installed
    (the build sandbox had no PyPI access). With the real package installed this does nothing."""
    try:
        import langchain_core  # noqa: F401
        return False
    except ImportError:
        pass

    class Runnable:
        def __or__(self, other):
            return Seq(self, other if isinstance(other, Runnable) else RunnableLambda(other))

    class Seq(Runnable):
        def __init__(self, a, b):
            self.a, self.b = a, b

        def invoke(self, x):
            return self.b.invoke(self.a.invoke(x))

    class RunnableLambda(Runnable):
        def __init__(self, f):
            self.f = f

        def invoke(self, x):
            return self.f(x)

    class Assign(Runnable):
        def __init__(self, **kw):
            self.kw = kw

        def invoke(self, x):
            return {**x, **{k: v.invoke(x) for k, v in self.kw.items()}}

    class RunnablePassthrough:
        assign = staticmethod(lambda **kw: Assign(**kw))

    class Prompt(Runnable):
        def __init__(self, msgs):
            self.msgs = msgs

        def invoke(self, vals):
            return "\n".join(t.format(**vals) for _, t in self.msgs)

    class ChatPromptTemplate:
        from_messages = staticmethod(lambda m: Prompt(m))

    class StrOutputParser(Runnable):
        def invoke(self, x):
            return getattr(x, "content", x)

    for name, attrs in {
        "langchain_core": {},
        "langchain_core.runnables": {"RunnableLambda": RunnableLambda, "RunnablePassthrough": RunnablePassthrough},
        "langchain_core.prompts": {"ChatPromptTemplate": ChatPromptTemplate},
        "langchain_core.output_parsers": {"StrOutputParser": StrOutputParser},
    }.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
    return True
