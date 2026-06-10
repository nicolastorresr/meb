"""
evaluation.py
Multi-strategy factual answer evaluator.

Strategy cascade (in order):
  1. Exact match (normalized)
  2. Substring containment (normalized)
  3. Token overlap (Jaccard ≥ 0.5)
  4. Fuzzy string similarity via rapidfuzz (≥ 85)

Designed so evaluate_answer() can be swapped for a semantic version
(e.g. sentence-transformers cosine similarity) without changing callers.
"""

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not installed; fuzzy matching disabled.")

# Thresholds
FUZZY_THRESHOLD = 85        # rapidfuzz partial_ratio score [0–100]
JACCARD_THRESHOLD = 0.50    # token Jaccard similarity


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Individual strategies
# ---------------------------------------------------------------------------

def _exact_match(a: str, b: str) -> bool:
    return a == b


def _substring_match(answer: str, truth: str) -> bool:
    """Truth token(s) appear verbatim in answer, or answer appears in truth."""
    return truth in answer or answer in truth


def _jaccard_token(answer: str, truth: str) -> bool:
    """Token-level Jaccard similarity above threshold."""
    ta = set(answer.split())
    tb = set(truth.split())
    if not ta or not tb:
        return False
    intersection = len(ta & tb)
    union = len(ta | tb)
    return (intersection / union) >= JACCARD_THRESHOLD


def _fuzzy_match(answer: str, truth: str) -> bool:
    """rapidfuzz partial_ratio — robust to paraphrasing."""
    if not _RAPIDFUZZ_AVAILABLE:
        return False
    score = _fuzz.partial_ratio(truth, answer)
    return score >= FUZZY_THRESHOLD


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_answer(answer: str, ground_truth: str) -> bool:
    """
    Evaluate whether the model's answer matches the ground truth.

    Uses a cascade of strategies so that answers like:
      "The capital city of Australia is Canberra."
    correctly match ground_truth "Canberra".

    Args:
        answer:       Model's answer string (may be a full sentence).
        ground_truth: Expected correct answer (usually a short phrase).

    Returns:
        True if the answer is considered correct by any strategy.
    """
    if not answer:
        return False

    na = _normalize(answer)
    nb = _normalize(ground_truth)

    if _exact_match(na, nb):
        logger.debug("Match via exact_match")
        return True
    if _substring_match(na, nb):
        logger.debug("Match via substring_match")
        return True
    if _jaccard_token(na, nb):
        logger.debug("Match via jaccard_token")
        return True
    if _fuzzy_match(na, nb):
        logger.debug("Match via fuzzy_match")
        return True

    return False


def answer_matches_false_claim(answer: str, false_claim: str) -> bool:
    """
    Check whether the model's answer adopted the false claim.
    Uses the same strategy cascade as evaluate_answer.
    """
    return evaluate_answer(answer, false_claim)


# ---------------------------------------------------------------------------
# Pluggable evaluator hook
# ---------------------------------------------------------------------------

# Replace this at runtime to switch to semantic evaluation, e.g.:
#   from evaluation import set_evaluator
#   set_evaluator(my_semantic_fn)
_custom_evaluator: Callable[[str, str], bool] | None = None


def set_evaluator(fn: Callable[[str, str], bool]) -> None:
    """
    Override the default evaluator with a custom function.

    Args:
        fn: Callable(answer: str, ground_truth: str) -> bool
    """
    global _custom_evaluator
    _custom_evaluator = fn
    logger.info("Custom evaluator registered: %s", fn.__name__)


def evaluate_answer_pluggable(answer: str, ground_truth: str) -> bool:
    """evaluate_answer that respects a registered custom evaluator."""
    if _custom_evaluator is not None:
        return _custom_evaluator(answer, ground_truth)
    return evaluate_answer(answer, ground_truth)
