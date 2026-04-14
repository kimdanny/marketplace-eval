from __future__ import annotations

from typing import Any, Dict, List, Sequence
import random

from marketplace_eval.system.types import UserProfile, UserQuery


class UserPopulation:
    """Utility class to sample synthetic user profiles and questions."""

    def __init__(
        self,
        *,
        profiles: Sequence[UserProfile],
        user_data: List[Dict[str, Any]],
        rng_seed: int | None = None,
    ):
        """
        Initialize user population with profiles and user data.

        Args:
            profiles: List of user profiles
            user_data: List of user data entries, each containing:
                - question (required): The question text
                - qid (optional): Question identifier for tracking
                - answer (optional): Ground truth answer for evaluation
                - context (optional): Context documents that ground the question
                - document_ids (optional): IDs of documents used to generate the question
            rng_seed: Random seed for reproducibility
        """
        self.profiles = list(profiles)
        self.user_data = user_data
        self.rng = random.Random(rng_seed)

        # Pre-compute sampling weights
        self._profile_weights = [p.sampling_probability for p in self.profiles]

        # Validate user data
        if not user_data:
            raise ValueError("user_data cannot be empty")
        for idx, entry in enumerate(user_data):
            if "question" not in entry:
                raise ValueError(
                    f"User data entry {idx} is missing required 'question' field"
                )

        # Without-replacement question sampling state
        self._remaining_indices: List[int] = []
        self._reset_question_pool()

    def _reset_question_pool(self):
        """Reset the question pool for without-replacement sampling."""
        self._remaining_indices = list(range(len(self.user_data)))
        self.rng.shuffle(self._remaining_indices)

    def sample_profiles(self, n: int) -> List[UserProfile]:
        """
        Sample n user profiles from the population based on their sampling probabilities.
        The same profile object may be sampled multiple times.
        Profiles maintain their state and learned preferences across samplings.
        """
        sampled_profiles = self.rng.choices(
            self.profiles, weights=self._profile_weights, k=n
        )
        return sampled_profiles

    def sample_question(self, profile: UserProfile) -> UserQuery:
        """
        Sample a question from the user data without replacement.

        Questions are drawn from a shuffled pool. Once all questions have been
        used, the pool is reset so every question is used at least once before
        any question is reused.

        Args:
            profile: The user profile to associate with the query
        """
        # If pool is empty, reset it
        if not self._remaining_indices:
            self._reset_question_pool()

        # Pop next index from the shuffled pool
        idx = self._remaining_indices.pop()
        data_entry = self.user_data[idx]

        # Extract fields from the data entry
        raw_question = data_entry["question"]
        qid = data_entry.get("qid")
        answer = data_entry.get("answer")
        context = data_entry.get("context")
        document_ids = data_entry.get("document_ids")

        return UserQuery(
            profile=profile,
            raw_question=raw_question,
            qid=qid,
            answer=answer,
            context=context,
            document_ids=document_ids,
        )
