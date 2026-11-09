"""A small BM25 retriever over guideline sections (pure Python, no dependencies).

This is lexical retrieval. The interface is deliberately tiny (search(query, k)) so an
embedding-based retriever can replace it later without touching the agent.
"""
import math
import re
from collections import Counter

from .guidelines import Section

TOKEN = re.compile(r"[a-z0-9çğıöşü]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, sections: list[Section], k1: float = 1.5, b: float = 0.75):
        self.sections = sections
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(f"{s.title} {s.title} {s.text}") for s in sections]  # the title counts double
        self.lengths = [len(d) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_freqs = [Counter(d) for d in self.docs]
        document_freq: Counter = Counter()
        for doc in self.docs:
            document_freq.update(set(doc))
        total = len(self.docs)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in document_freq.items()
        }

    def search(self, query: str, k: int = 3) -> list[tuple[Section, float]]:
        terms = tokenize(query)
        scored = []
        for index, section in enumerate(self.sections):
            score = 0.0
            frequencies = self.term_freqs[index]
            for term in terms:
                if term not in frequencies:
                    continue
                freq = frequencies[term]
                norm = 1 - self.b + self.b * self.lengths[index] / (self.avg_length or 1)
                score += self.idf[term] * freq * (self.k1 + 1) / (freq + self.k1 * norm)
            if score > 0:
                scored.append((section, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
