"""The evaluation harness: efficiency, precision/recall, retention, transfer.

Oracle-side code -- see `eval.py`'s module docstring for why that matters and
for the concept-to-kind resolution problem this module exists to solve
correctly.
"""

from .eval import ConceptKindTracker, ConceptResolution, Evaluator

__all__ = ["ConceptKindTracker", "ConceptResolution", "Evaluator"]
