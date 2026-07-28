"""Verification: deciding what to test, testing it, and folding the result
back into memory.

`select.py` ranks hypotheses by how much a real interaction there would still
teach the bank, per unit of budget spent. `loop.py` turns a choice into an
actual environment step and an `AffordanceMemory.observe` call, closing
`observe -> imagine -> select -> verify -> adjudicate -> remember`.
"""

from .loop import VerificationStep, run_episode, verify_once
from .select import rank, select_next, uncertainty

__all__ = [
    "VerificationStep",
    "rank",
    "run_episode",
    "select_next",
    "uncertainty",
    "verify_once",
]
