# Project decisions

A running record of choices that shape the codebase, with the reasoning behind
them. Append new entries; do not rewrite old ones. If a decision is reversed,
add a new entry that says so and why.

---

## D1. LAMA supersedes AURA as the active research line

**Decision.** LAMA (Lifelong Affordance learning through counterfactual
verificAtion) is the project this repository now builds. The AURA project plan
(`AURA_Project_Plan.pdf`) and the AURA warehouse prototype
(`aura_warehouse_sim/`) are retained unchanged as prior work. All new code lives
under `lama/`.

**Why.** AURA's research question was "can a world model detect the situation
types where it is unreliable and collect targeted experience to repair them".
LAMA's is narrower and more testable: "can an agent decide which of its imagined
affordance hypotheses are worth testing in reality, and accumulate only verified
knowledge". The self-improvement machinery AURA describes is a superset that was
never built; LAMA is the part with a crisp, measurable claim.

**Not done.** The repository name and git remote are unchanged. Renaming them is
a separate, owner-level decision.

---

## D2. A new purpose-built affordance environment, not the existing simulators

**Decision.** Build a new environment under `lama/env/`: a fast, numpy-only
warehouse with typed objects that carry a *hidden affordance profile*, and
object-directed actions. Do not extend `aura_warehouse_sim/sim.py`, and do not
vendor the pymunk environment from `files/EmbodiedAI.zip`.

**Why.**

1. *Interaction volume is the binding constraint.* The prior EmbodiedAI project
   reached an explicit negative result: no function-learning objective beat
   random initialisation at 186 interaction records over 13 objects, and its own
   conclusion was "the bottleneck is interaction volume, not architecture".
   LAMA needs 10^4-10^5 interactions to say anything. A rigid-body simulator
   that renders 480x480 through PIL on every step cannot supply them on CPU.
2. *Evaluation needs an explicit ground-truth affordance table.* LAMA is scored
   on "confirmed knowledge that is actually correct" and "interactions spent per
   confirmed affordance". That requires the environment to know, exactly, which
   (object, action) pairs afford what. Rigid-body physics only exposes this
   indirectly through displacement.
3. *Physical fidelity does not serve the claim.* The research object is the
   verification loop, not contact dynamics. Fidelity adds cost without
   strengthening the result.
4. `aura_warehouse_sim/sim.py` has navigation only. It has no object-interaction
   model at all, so extending it is the same work as writing a new environment
   while inheriting constraints chosen for a different purpose.
5. The pymunk environment additionally costs a dependency (`pymunk` is not
   installed here), and its five benchmark levels test *detour planning*
   (fetch key, unlock door), which is a different problem from affordance
   verification.

**Design commitments that follow.**

- Affordance is a hidden property of `(object kind, action, context)`, known to
  the environment and never observable by the agent.
- Some outcomes are genuinely stochastic. This is deliberate: LAMA must separate
  epistemic uncertainty (missing knowledge, worth verifying) from aleatoric
  uncertainty (irreducible randomness, not worth verifying). An agent that
  cannot tell them apart will waste its interaction budget, and that is exactly
  what the evaluation should expose.
- Appearance correlates with affordance but does not determine it. Objects that
  look alike may behave differently, so the agent cannot shortcut verification
  by memorising colour.
- Held-out object kinds exist for the transfer evaluation.

**Reused from prior work (rewritten, not copied).** The structured
interaction-record schema and SQLite/vector-index memory pattern from
`memory/experience.py`; the causal transformer world-model shape from
`world_model/transformer.py`; and the practice of reporting negative results
plainly.

---

## D3. Baseline of the working tree

**Decision.** Treat commit `f6cac28` as the starting point.

**Why.** The owner reported having made changes, but the working tree was clean
and every file present belonged to that single commit, so no separate changes
could be identified. Recorded here so the assumption is visible rather than
silent.

---

## D4. Isaac Lab is a later backend, so the environment interface is a contract

**Decision.** Isaac Lab is planned as a second environment backend in a later
phase of the project. It is not a dependency now. Consequently, `lama/env/`
defines an abstract environment interface first, and the fast numpy warehouse
from D2 is one implementation of it. An `IsaacWarehouse` will be another.

**Why.** The point of building the cheap environment first is that the expensive
one can arrive later without invalidating the work. That only holds if nothing
downstream — encoder, world model, imagination, verification, affordance bank,
evaluation — ever touches a backend-specific detail. If those modules are
written against a concrete numpy environment, the Isaac Lab migration becomes a
rewrite instead of a swap. Fixing the interface now costs one file; fixing it
later costs the codebase.

**What this constrains.**

- Everything downstream imports the interface, never a concrete environment.
- Observations, actions and interaction records are typed structures defined by
  the interface, not whatever the numpy backend happens to return.
- Continuous quantities are used wherever a physics backend would produce them,
  so the numpy backend does not bake in grid assumptions that PhysX cannot
  reproduce.
- Hidden ground-truth affordance queries, used only by evaluation, sit behind a
  clearly marked and separately implementable part of the interface. A backend
  that cannot answer them can still run the agent; it just cannot score it.
- The backend is chosen by configuration, and the evaluation harness records
  which backend produced every result.

**Open question, deferred.** Isaac Lab needs a CUDA-capable NVIDIA GPU. That
constraint is not yet confirmed for this project, and should be settled before
the migration phase starts rather than during it.

---

## D5. Affordance taxonomy: a verb set, and primary versus secondary use

**Decision.** Objects are understood through a fixed set of interaction verbs —
`approach, push, pull, lift, press, rotate, open, close, grasp, release,
place_on, tip` — and every `(object kind, verb)` pair carries a hidden
*affordance role*:

| Role | Meaning | Example |
|---|---|---|
| `PRIMARY` | what the object is for; its canonical function | `press` a button, `open` a door, `push` a cart |
| `SECONDARY` | a real but non-canonical use; the object as a means to another end | `place_on` a crate onto a pressure plate to hold it down; `lift` a stool to reach a high shelf |
| `INCIDENTAL` | works, but carries little information; nearly everything affords it weakly | nudging a shelf a centimetre |
| `NONE` | the object does not afford this verb | `lift` a wall |

**Why this is the core of the project.** The interesting claim is not "the agent
learns objects can be pushed". It is that **canonical uses are easy to stumble
into and improvised uses are not**. Random and curiosity-driven exploration find
`PRIMARY` affordances readily, because they are what the object responds to most
strongly and most often. `SECONDARY` affordances are rare, frequently
conditional, and often only meaningful through their effect on *something else*
in the world. If counterfactual verification has an advantage, this is where it
shows up, so the evaluation must score `PRIMARY` and `SECONDARY` discovery
separately. A single averaged number would hide the entire result.

**Consequences for the environment.**

1. **Remote effects are first-class.** An outcome may change a *different*
   object than the one acted on (a weight on a plate opens a door across the
   room). Without this there is no way to express "the crate's other use", and
   secondary affordance collapses into a synonym for primary.
2. **Relational verbs are first-class.** `place_on` takes two objects. Tool use
   is inherently relational; a verb set of unary actions cannot represent it.
3. **Preconditions exist.** Some affordances only hold in context — gripper
   free, object already held, object already open. The agent must discover the
   precondition, not just the verb.
4. **Some outcomes are irreversible.** A verification that breaks an object, or
   consumes it, cannot be undone. This gives hypothesis selection real stakes:
   the cost of a test is not uniform, and an agent that ignores this will
   destroy the evidence it needed.
5. **Reliability is explicit.** Each true affordance carries a probability. An
   affordance that fires 60% of the time is not the same as one the agent simply
   has not learned yet, and separating those two is the hard part of selection.

**On "instinct".** The agent should behave as though it has an intuition about
what an object is for before touching it — the world model supplies exactly
that: a *prior over affordances conditioned on appearance*. The loop is then
instinct, doubt, test, knowledge. Appearance is therefore designed to be
informative but not decisive: objects that look alike may behave differently, so
the prior is a real hypothesis rather than a lookup, and verification has
something to do.

*Reading of the request:* "social instinct" was taken to mean this common-sense
prior about object use. If it instead meant affordances involving other agents
(learning by watching a human use an object), that is a different and larger
feature; it is not built, and should get its own decision entry.

**Consequences for evaluation.** The affordance bank is scored against hidden
ground truth on precision and recall, broken out by role. Interactions spent per
confirmed `SECONDARY` affordance is the headline efficiency number, because that
is the quantity the baselines should struggle with.
