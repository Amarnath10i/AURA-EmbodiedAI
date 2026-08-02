"""Goal-directed planning: backward chaining over confirmed knowledge.

Answers "which of the currently-worth-trying hypotheses are actually on the
way to some goal", not "what should the agent do right now" -- that stays
`verification/select.py`'s job, now goal-aware via
`RegressionPlanner.relevant_keys`.
"""

from .planner import Goal, Operator, PlanStep, RegressionPlanner, operator_from_belief

__all__ = [
    "Goal",
    "Operator",
    "PlanStep",
    "RegressionPlanner",
    "operator_from_belief",
]
