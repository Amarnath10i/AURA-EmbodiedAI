"""Deciding where to walk when nothing reachable is worth testing yet."""

from .active import DEFAULT_DISTANCE_WEIGHT, ExplorationTarget, select_exploration_target

__all__ = ["DEFAULT_DISTANCE_WEIGHT", "ExplorationTarget", "select_exploration_target"]
