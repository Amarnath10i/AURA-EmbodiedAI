"""Interaction memory: the raw ledger and the lifelong affordance bank.

Nothing here ever touches hidden ground truth. Everything is built from what
`lama.env` actually hands the agent -- appearance descriptors, opaque object
ids, outcomes -- which is what makes this the place where the crate/block
trap becomes a real cost rather than a decoration. See `concepts.py` and
`bank.py` for how.
"""
