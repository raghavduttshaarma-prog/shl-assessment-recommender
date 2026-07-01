"""
Catalogue loader and search engine for SHL assessments.

Uses TF-IDF for semantic matching + keyword/faceted filtering.
"""

import json
import os
import re
import math
from typing import List, Dict, Optional, Tuple
from collections import Counter


# ──────────────────────────────────────────────
# Category → test_type mapping
# ──────────────────────────────────────────────
CATEGORY_TO_TYPE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def get_test_type(keys: List[str]) -> str:
    """Map catalogue 'keys' to SHL's single-letter test_type code(s).

    An assessment can belong to multiple categories (e.g. a bundled
    solution that is both a Knowledge test and a Simulation), in which
    case the codes are joined with a comma in the order the categories
    appear in the source data, e.g. "K,S". Falls back to "K" if none
    of the keys are recognized.
    """
    codes = []
    for cat in keys:
        code = CATEGORY_TO_TYPE.get(cat)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


class Assessment:
    """Represents a single SHL assessment product."""

    def __init__(self, raw: dict):
        self.entity_id = raw.get("entity_id", "")
        self.name = raw.get("name", "")
        self.link = raw.get("link", "")
        self.description = raw.get("description", "")
        self.job_levels = raw.get("job_levels", [])
        self.languages = raw.get("languages", [])
        self.duration = raw.get("duration", "")
        self.remote = raw.get("remote", "no")
        self.adaptive = raw.get("adaptive", "no")
        self.keys = raw.get("keys", [])
        self.test_type = get_test_type(self.keys)

        # Build searchable text blob
        self.search_text = " ".join([
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(self.job_levels),
        ]).lower()

    def to_recommendation(self) -> dict:
        return {
            "name": self.name,
            "url": self.link,
            "test_type": self.test_type,
        }

    def to_context_string(self) -> str:
        """Return a compact one-item text summary for LLM context.

        Kept deliberately terse (~40-60 tokens/item) because this string is
        repeated for every candidate on every single stateless /chat call —
        free-tier LLM token budgets (e.g. Groq's 100k tokens/day) get eaten
        alive fast if each candidate costs 150+ tokens.
        """
        levels = ",".join(self.job_levels[:3]) + ("+" if len(self.job_levels) > 3 else "") if self.job_levels else "N/A"
        langs = ",".join(self.languages[:2]) + ("+" if len(self.languages) > 2 else "") if self.languages else "N/A"
        desc = self.description[:100].rsplit(" ", 1)[0] if len(self.description) > 100 else self.description
        return (
            f"[{self.entity_id}] {self.name} | {self.test_type} | {levels} | "
            f"{self.duration or 'N/A'} | remote={self.remote} adaptive={self.adaptive} | {langs} | {self.link}\n"
            f"  {desc}\n"
        )


class CatalogSearchEngine:
    """TF-IDF + faceted search over SHL assessment catalogue."""

    def __init__(self):
        self.assessments: List[Assessment] = []
        self.name_index: Dict[str, Assessment] = {}  # lowercase name → Assessment
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: List[Dict[str, float]] = []

    def load(self, path: str = None):
        """Load catalogue from JSON file."""
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "catalogue.json"
            )


        with open(path, "r", encoding="utf-8-sig") as f:
            raw_data = json.load(f)

        self.assessments = [Assessment(item) for item in raw_data]
        self.name_index = {a.name.lower(): a for a in self.assessments}

        # Build TF-IDF index
        self._build_tfidf()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'[a-z0-9#+.]+', text.lower())

    def _build_tfidf(self):
        """Build TF-IDF vectors for all assessments."""
        n = len(self.assessments)
        doc_freq: Counter = Counter()

        # Tokenize all docs
        all_tokens = []
        for a in self.assessments:
            tokens = self._tokenize(a.search_text)
            all_tokens.append(tokens)
            unique = set(tokens)
            for t in unique:
                doc_freq[t] += 1

        # Compute IDF
        self.idf = {
            term: math.log(n / (df + 1))
            for term, df in doc_freq.items()
        }

        # Compute TF-IDF vectors
        self.tfidf_vectors = []
        for tokens in all_tokens:
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec = {
                term: (count / total) * self.idf.get(term, 0)
                for term, count in tf.items()
            }
            self.tfidf_vectors.append(vec)

    def _query_vector(self, query: str) -> Dict[str, float]:
        """Convert a query to a TF-IDF vector."""
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {
            term: (count / total) * self.idf.get(term, 0)
            for term, count in tf.items()
        }

    def _cosine_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def search(
        self,
        query: str,
        top_k: int = 15,
        category_filter: Optional[List[str]] = None,
        job_level_filter: Optional[str] = None,
    ) -> List[Tuple[Assessment, float]]:
        """
        Hybrid search: TF-IDF similarity + faceted bonuses.

        Returns list of (Assessment, score) tuples sorted by descending score.
        """
        query_vec = self._query_vector(query)
        query_lower = query.lower()
        results = []

        for i, assessment in enumerate(self.assessments):
            # Base TF-IDF score
            score = self._cosine_similarity(query_vec, self.tfidf_vectors[i])

            # Name match bonus
            if any(word in assessment.name.lower() for word in query_lower.split() if len(word) > 2):
                score += 0.3

            # Exact name substring match bonus
            if query_lower in assessment.name.lower():
                score += 0.5

            # Category filter bonus/penalty
            if category_filter:
                cat_match = any(
                    cat.lower() in [k.lower() for k in assessment.keys]
                    for cat in category_filter
                )
                if cat_match:
                    score += 0.2
                else:
                    score *= 0.5

            # Job level filter bonus
            if job_level_filter:
                level_lower = job_level_filter.lower()
                if any(level_lower in jl.lower() for jl in assessment.job_levels):
                    score += 0.15

            if score > 0.01:
                results.append((assessment, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_by_name(self, name: str) -> Optional[Assessment]:
        """Look up an assessment by exact name (case-insensitive)."""
        return self.name_index.get(name.lower())

    def get_all_categories(self) -> List[str]:
        """Return all unique categories."""
        cats = set()
        for a in self.assessments:
            cats.update(a.keys)
        return sorted(cats)

    def get_assessments_for_context(self, query: str, top_k: int = 15) -> str:
        """Get formatted assessment context for LLM."""
        results = self.search(query, top_k=top_k)
        if not results:
            return "No matching assessments found."

        lines = [f"Found {len(results)} relevant assessments:\n"]
        for assessment, score in results:
            lines.append(assessment.to_context_string())
        return "\n".join(lines)

    def validate_recommendation(self, name: str) -> Optional[Assessment]:
        """Validate that a recommendation exists in the catalogue."""
        # Try exact match
        result = self.get_by_name(name)
        if result:
            return result

        # Try fuzzy match (contains), but require enough signal to avoid
        # short substrings (e.g. "SQL") matching unrelated long names.
        name_lower = name.lower().strip()
        if len(name_lower) < 4:
            return None

        best, best_len = None, 0
        for a in self.assessments:
            a_lower = a.name.lower()
            if name_lower in a_lower or a_lower in name_lower:
                # Prefer the longest / most specific overlapping match.
                if len(a_lower) > best_len:
                    best, best_len = a, len(a_lower)

        return best

    def find_by_substring(self, text: str, limit: int = 8) -> List["Assessment"]:
        """Find assessments explicitly named/referenced inside free text.

        Used to ground COMPARE and follow-up questions ("what's the
        difference between X and Y") to the exact catalogue entries being
        discussed, rather than relying purely on TF-IDF similarity.
        """
        text_lower = text.lower()
        matches = []
        for a in self.assessments:
            name_lower = a.name.lower()
            base = re.sub(r'\s*\(new\)\s*$', '', name_lower).strip()
            if base and (base in text_lower or name_lower in text_lower):
                matches.append(a)
                continue
            for tok in re.findall(r'[a-z0-9]+', base):
                if len(tok) >= 3 and tok not in _STOPWORDS and re.search(rf'\b{re.escape(tok)}\b', text_lower):
                    matches.append(a)
                    break

        seen = set()
        unique = []
        for a in sorted(matches, key=lambda x: -len(x.name)):
            if a.entity_id not in seen:
                seen.add(a.entity_id)
                unique.append(a)
        return unique[:limit]


# Words too generic to count as a meaningful name-fragment match.
_STOPWORDS = {
    "new", "test", "tests", "the", "and", "for", "with", "report", "level",
    "general", "advanced", "entry", "basic", "development", "assessment",
    "solution", "focus", "individual", "professional", "manager", "sales",
    "customer", "service", "essentials", "interactive", "verify",
}


# Singleton instance
catalog = CatalogSearchEngine()
