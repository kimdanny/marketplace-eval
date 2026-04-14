"""
Automatic generation of user profiles.

Profiles are simple: N users with uniform generator preferences and equal
sampling probability.  Neither question types nor user types are attached to
profiles.  Both are sampled independently at data-generation time based on
the taxonomy's probability distributions.
"""

from __future__ import annotations

from typing import List

from marketplace_eval.system.types import UserProfile


def generate_profiles(
    generator_ids: List[str],
    total_num_users: int = 10,
) -> List[UserProfile]:
    """
    Generate user profiles with uniform preferences across generators.

    Args:
        generator_ids: List of generator node IDs to initialise preferences.
        total_num_users: Number of user profiles to generate.

    Returns:
        List of UserProfile objects with equal sampling probabilities.
    """
    equal_prob = 1.0 / len(generator_ids) if generator_ids else 1.0
    initial_preferences = {gen_id: equal_prob for gen_id in generator_ids}
    sampling_prob = 1.0 / total_num_users

    profiles: List[UserProfile] = []
    for i in range(total_num_users):
        profile = UserProfile(
            user_id=f"user_{i + 1}",
            preference_scores=initial_preferences.copy(),
            sampling_probability=sampling_prob,
            rng_state={},
        )
        profiles.append(profile)

    return profiles
