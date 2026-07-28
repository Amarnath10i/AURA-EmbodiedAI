"""Verification: deciding what to test, testing it, and folding the result
back into memory.

`select.py` ranks hypotheses by how much a real interaction there would still
teach the bank, per unit of budget spent.
"""

from .select import rank, select_next, uncertainty

__all__ = ["rank", "select_next", "uncertainty"]
