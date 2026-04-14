from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import random


@dataclass
class UserProfile:
    user_id: str
    preference_scores: Dict[str, float]  # generator preferences
    question_style: str = ""  # optional: used for manual profile config
    question_domain: str = ""  # optional: reserved for future use
    sampling_probability: float = 1.0  # probability of sampling this profile
    rng_state: Any = field(default_factory=dict)

    def choose_generator(
        self,
        rng: random.Random,
        epsilon: float = 0.2,
        available_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Epsilon-greedy selection: with probability 1-epsilon choose the
        generator with highest preference; with probability epsilon choose
        uniformly at random. If available_ids is set, only generators in that
        list can be chosen (e.g. generators introduced by the current timestep).
        """
        generators = list(self.preference_scores.keys())
        if available_ids is not None:
            generators = [g for g in generators if g in available_ids]
            if not generators:
                generators = list(available_ids)
        if not generators:
            raise ValueError(
                "Preference scores must be non-empty or available_ids must be non-empty"
            )

        if rng.random() < epsilon:
            return rng.choice(generators)
        best = max(
            generators,
            key=lambda g: self.preference_scores.get(g, 0.0),
        )
        return best

    def add_generator(self, generator_id: str):
        """Add a newly introduced generator with a fair initial preference.

        The new generator receives preference equal to 1/(n+1) where n is the
        current number of generators.  Existing preferences are scaled down
        proportionally so that relative ordering is preserved and the
        distribution still sums to 1.
        """
        if generator_id in self.preference_scores:
            return  # already present
        n = len(self.preference_scores)
        if n == 0:
            self.preference_scores[generator_id] = 1.0
            return
        new_share = 1.0 / (n + 1)
        scale = n / (n + 1)  # existing preferences scaled to sum to n/(n+1)
        for key in self.preference_scores:
            self.preference_scores[key] *= scale
        self.preference_scores[generator_id] = new_share

    def update_preference(
        self, generator_id: str, score: float, drift_rate: float = 0.1
    ):
        base = self.preference_scores.get(generator_id, 0.01)
        self.preference_scores[generator_id] = max(0.01, base + drift_rate * score)
        total = sum(self.preference_scores.values())
        # converting to probability distribution
        if total > 0:
            for key in list(self.preference_scores.keys()):
                self.preference_scores[key] = self.preference_scores[key] / total


@dataclass
class UserQuery:
    profile: UserProfile
    raw_question: str
    qid: str | None = None  # Question ID for tracking
    answer: str | None = None  # Ground truth answer for evaluation (when available)
    context: List[str] | None = None  # Context/documents that ground the question
    document_ids: List[
        str
    ] | None = None  # IDs of documents used to generate the question


@dataclass
class RetrievalCall:
    retriever_id: str
    documents: List[str]
    scores: List[float]
    router_id: str | None = None


@dataclass
class GenerationResult:
    user_id: str
    generator_id: str
    question: str
    answer: str
    retrievals: List[RetrievalCall]
    metadata: Dict[str, Any]


@dataclass
class JudgeFeedback:
    score: float
    rationale: str
    retriever_scores: Dict[str, float]
    generator_feedback: Dict[str, Any] = field(default_factory=dict)
