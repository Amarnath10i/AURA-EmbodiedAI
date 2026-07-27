"""LAMA — Lifelong Affordance learning through counterfactual verificAtion.

An embodied agent that imagines candidate interactions inside a world model,
spends real interactions only on the hypotheses it is most uncertain about, and
keeps only what verification confirms.

Package layout (built incrementally; see docs/DECISIONS.md):

    env/            environment interface + backends (numpy now, Isaac Lab later)
    perception/     observation -> object-centric latents
    world_model/    outcome prediction with calibrated uncertainty
    imagination/    counterfactual rollouts over candidate interactions
    verification/   hypothesis selection, execution and adjudication
    memory/         raw interaction log + the lifelong affordance bank
    planner/        task planning over confirmed knowledge
    evaluation/     metrics, baselines and the reporting harness
"""

__version__ = "0.1.0.dev0"
