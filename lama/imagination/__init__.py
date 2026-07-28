"""Counterfactual imagination: candidate interactions, with a prediction.

`imagine` turns "what can I do right now" into `Hypothesis` objects carrying
whatever the affordance bank already believes about each. `verification/`
decides which of them are worth spending real budget to test.
"""

from .hypothesis import HYPOTHESIS_ACTIONS, Hypothesis, imagine

__all__ = ["HYPOTHESIS_ACTIONS", "Hypothesis", "imagine"]
