"""
dataset_loader.py
Loads and validates the questions dataset and pressure templates.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Represents a factual question with metadata."""

    id: str
    category: str
    question: str
    ground_truth: str
    false_claim: str

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        required = {"id", "category", "question", "ground_truth", "false_claim"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Question missing fields: {missing}")
        return cls(
            id=data["id"],
            category=data["category"],
            question=data["question"],
            ground_truth=data["ground_truth"],
            false_claim=data["false_claim"],
        )


class DatasetLoader:
    """Loads questions and pressure templates from JSON files."""

    PRESSURE_TYPES = ["authority", "majority", "emotional", "iterative_contradiction"]

    def __init__(
        self,
        questions_path: str | Path,
        templates_path: str | Path,
        seed: Optional[int] = 42,
    ):
        self.questions_path = Path(questions_path)
        self.templates_path = Path(templates_path)
        self.rng = random.Random(seed)
        self._questions: list[Question] = []
        self._templates: dict[str, list[str]] = {}

    def load(self) -> None:
        """Load both datasets from disk."""
        self._load_questions()
        self._load_templates()
        logger.info(
            "Loaded %d questions and %d pressure types.",
            len(self._questions),
            len(self._templates),
        )

    def _load_questions(self) -> None:
        with open(self.questions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._questions = [Question.from_dict(q) for q in raw]
        if not self._questions:
            raise ValueError("Questions dataset is empty.")

    def _load_templates(self) -> None:
        with open(self.templates_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for pt in self.PRESSURE_TYPES:
            if pt not in raw:
                raise ValueError(f"Pressure type '{pt}' missing from templates.")
            if not raw[pt]:
                raise ValueError(f"Pressure type '{pt}' has no templates.")
        self._templates = raw

    @property
    def questions(self) -> list[Question]:
        if not self._questions:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return list(self._questions)

    @property
    def pressure_types(self) -> list[str]:
        return list(self.PRESSURE_TYPES)

    def get_pressure_prompt(self, pressure_type: str, false_claim: str) -> str:
        """
        Return a randomly selected pressure prompt for the given type,
        formatted with the false claim.

        Args:
            pressure_type: One of the PRESSURE_TYPES.
            false_claim: The incorrect claim to inject.

        Returns:
            Formatted pressure string.
        """
        if pressure_type not in self._templates:
            raise ValueError(f"Unknown pressure type: {pressure_type}")
        template = self.rng.choice(self._templates[pressure_type])
        return template.format(false_claim=false_claim)

    def get_pressure_sequence(
        self, pressure_type: str, false_claim: str, n: int = 3
    ) -> list[str]:
        """
        Return a sequence of n distinct (if possible) pressure prompts.

        Args:
            pressure_type: One of the PRESSURE_TYPES.
            false_claim: The incorrect claim to inject.
            n: Number of prompts to generate.

        Returns:
            List of formatted pressure strings.
        """
        templates = self._templates[pressure_type]
        if len(templates) >= n:
            chosen = self.rng.sample(templates, n)
        else:
            chosen = [self.rng.choice(templates) for _ in range(n)]
        return [t.format(false_claim=false_claim) for t in chosen]
