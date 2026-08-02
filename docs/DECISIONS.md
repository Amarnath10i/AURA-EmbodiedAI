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

---

## D6. Isaac Lab backend: GPU question resolved, written unverified

**Decision.** D4's open question -- whether GPU access for Isaac Lab was
confirmed -- is now answered. The development machine (GTX 1650, 4GB VRAM)
cannot run Isaac Sim at all: no RT cores, VRAM below every generation's
minimum. The target machine (RTX 4060, 8GB VRAM, i7 12th-gen) clears the
minimum for the Isaac Sim 4.5.0 / Isaac Lab 2.1.0 generation (RTX 3070, 8GB)
but not the current `main`-branch minimum (RTX 4080, 16GB), which has climbed
release over release. `lama/env/isaac_warehouse.py` and
`docs/ISAAC_LAB_SETUP.md` pin to the 2.1.0/4.5.0 generation specifically
because of this gap -- installing "latest" on this hardware would very likely
fail outright.

Given that, the backend was built anyway, on the machine without a capable
GPU, entirely from current documentation, and has never been executed
against real Isaac Sim.

**Why build it unverified rather than wait.** The owner has access to the
target machine and asked for the code and tooling to exist so it can be
cloned and run there. Waiting would mean delivering nothing; building it
honestly-labelled as unverified, with the verification that *is* possible
done thoroughly, delivers a real head start and is falsifiable the moment it
reaches real hardware.

**How it was verified without the hardware.** `tests/test_isaac_warehouse_logic.py`
injects a fake `isaaclab` package into `sys.modules`, built from real `torch`
tensors rather than permissive mocks -- shapes and call signatures are
actually checked. This exercises every line of `IsaacWarehouse`: every method
call, every attribute access, the full affordance-resolution logic
(preconditions, mechanism latching, momentary door release), and the
flagship crate/block secondary affordance end to end. It caught one real bug
before commit: `_body_pos` returned a numpy view aliasing the same memory as
the live position tensor, so a before/after displacement snapshot silently
read the *same* value twice, making every measured push distance zero. What
this cannot verify is whether the fake's guessed signature for
`RigidObject.set_external_force_and_torque` -- the one call flagged as
highest-risk in the module's own docstring -- matches the real, currently
installed Isaac Lab version. Only real hardware settles that.

**Scope cuts, and why each is defensible.**

- *Grasping and lifting are state bookkeeping, not physical.* There is no
  articulated gripper anywhere in this project's agent model, on either
  backend -- `warehouse.py` already made this simplification (D2), and the
  Isaac backend mirrors it rather than introducing asymmetric fidelity
  between backends for a capability neither uses.
- *Doors do not physically swing.* `is_open`/`locked` are python state on
  both backends. The research question is whether a mechanism's *effect* is
  discoverable, not whether it looks like it swings; `ArticulationCfg` with a
  revolute joint is a clean future addition if that ever matters.
- *No collision-blocked navigation.* `APPROACH` teleports the agent on both
  backends (D2's original navigation-is-solved reasoning applies unchanged).
- *No camera rendering.* Both backends build `ObjectView.appearance` from the
  same `appearance.describe` synthetic descriptor. No perception encoder
  exists yet in this project (see the README status table), so rendering
  real frames now would add risk and complexity in service of a consumer
  that does not exist. Swapping in a real camera capture later is confined to
  one call site in `IsaacWarehouse._view`.
- *Layout composition is fixed per instance.* Isaac Sim can only be launched
  once per process, so `IsaacWarehouse` uses a module-level singleton and
  constructing a fresh instance per episode (the numpy backend's usual
  pattern) is not an option. Re-randomising which kinds are spawned on
  `reset()` would need runtime prim deletion; no clearly-documented pattern
  for that surfaced during research done offline from real hardware, so
  `reset()` only re-randomises dynamics (door states, agent spawn point) by
  teleporting the same spawned objects back to their initial poses. A
  different `layout_seed` means constructing a new instance. Documented as a
  real, temporary gap, not silently matched to the numpy backend's semantics.
- *Affordance-resolution logic is duplicated from `warehouse.py`, not
  shared.* A deeper refactor (the same kind of extraction `layout.py` did for
  kind-planning logic) was considered and rejected here specifically: a
  shared mixin is itself new,
  unverifiable surface area, and duplicating already-tested logic
  line-for-line is lower risk than restructuring it blind. Flagged as a
  future cleanup candidate once the Isaac backend has run for real and any
  necessary fixes have landed -- refactoring before that would risk
  reintroducing exactly the bugs the duplication was meant to avoid.

**What "done" looks like next.** Someone runs `scripts/isaac_smoke_test.py`
on the RTX 4060 machine, reports what breaks, and it gets fixed from a real
traceback instead of more documentation research.

---

## D7. Reconciling substantial untracked upstream work; the combinatorics,
disambiguation, and safety problems

**Context.** Between sessions, the repository was cloned onto the target RTX
4060 machine and, separately from this assistant, ten more commits were
pushed directly to the remote (also renamed AURA-EmbodiedAI to
LAMA-EmbodiedAI on GitHub in the process): a substantial batch implementing
"LAMA Phases 0-6" plus SOTA-labelled components -- CEM/MPPI planners, active
exploration, a DreamerV3-style world model, curiosity modules, an evaluation
harness -- together with real experiment output (outputs/) from actual runs
against both the numpy and Isaac Lab backends. The local clone was a strict
ancestor of the new remote head, so reconciling was a plain fast-forward, no
merge conflict.

**That work was reviewed in full before anything was built on top of it**,
because roughly 1300 lines of new functionality had shipped with zero new
test coverage. The review found:

- exploration/active.py could not be imported at all (it tried to import a
  ConceptMemory class that does not exist -- the real class is
  ConceptCodebook) and, even fixed, called Environment.step() with the
  wrong signature in three places.
- MPPIPlanner.plan() referenced self.discount, which MPPIPlanner.__init__
  never set -- a guaranteed crash the first time it ran.
- The composite-hypothesis code in hypothesis.py built Hypothesis objects
  with a synthetic target_id like "obj_03_obj_07" that names no real
  object; env.step() raises KeyError on it. It also had a real collision
  risk against genuine concept ids (hash((a, b)) % 100000) and generated
  each pair twice, asymmetrically.
- AffordanceBank.__init__ unconditionally opened and appended to
  outputs/bank_history.jsonl on every single .observe() call -- a
  relative-path disk write on the hot path, at the 10^4-10^5 interaction
  scale this project targets.
- select.py's own, separately-declared settled-status set omitted the new
  STUCK status, so a belief the bank had just flagged as probably blending
  two kinds scored as maximally worth testing instead of being
  deprioritised -- the opposite of the intended effect.
- evaluation/eval.py called env.oracle (neither backend has this
  attribute) and oracle.ground_truth_affordances() (no such method), and
  compared belief.key (agent concept ids) directly against ground-truth
  keys (kind names) -- two incompatible key spaces that could never match,
  so precision/recall would have silently been meaningless even with the
  crashes fixed.
- Two separate, redundant, untrained DreamerV3-style RSSM implementations
  (models/world_model.py, world_model/rssm.py) and a curiosity module
  (ICM/RND/contrastive) with zero consumers anywhere, targeting a 96x96
  image observation space this project does not use.

**Decision: keep and substantially develop the pieces that were genuinely
aligned with the project and the owner's stated goals; remove the rest
outright rather than patch around it.** Removed: both duplicate world
models, curiosity, CEMPlanner/MPPIPlanner, the broken composite-hypothesis
code, and the unconditional disk logging. Kept and rebuilt: RegressionPlanner
(the one genuinely classical-AI, backward-chaining piece, which is also
exactly what was asked for), active exploration, and the evaluation harness.
Reasoning: a disconnected, untested, crash-prone module produces negative
value just by existing in the tree -- it looks like coverage of a capability
that is not actually there, which is worse than the capability being
visibly absent from the README's status table.

**The three problems the owner asked to be solved.**

1. Combinatorial verb search. planner/planner.py's RegressionPlanner now
   does real STRIPS-style backward chaining: operators carry genuine
   preconditions (a verb's public needs_free_gripper/needs_held_object, not
   a placeholder), so chaining discovers real multi-step structure --
   verified against the project's own flagship case, where reaching the
   goal requires GRASPing the block before PLACE_ON can work, and the
   planner finds that ordering. relevant_keys() feeds a
   (concept, verb, tool) relevance weight into Hypothesis.relevance, which
   select.py now folds in multiplicatively. This does not shrink the
   hypothesis space by refusing options; it reorders it so the search tries
   goal-relevant combinations first, which is what actually matters when the
   verb set keeps growing. Left unset, a goal changes nothing --
   uncertainty-driven exploration is the default, goals are additive.

2. Objects that look identical but are not (the owner's "hollow vs solid
   sphere" example; this project's existing embodiment is the crate/block
   trap). Outcome.force_required is now populated by both backends -- only
   for the TRANSLATED effect specifically, since PLACE_ON's "displacement"
   is geometric distance carried, not mass-correlated, and feeding it in
   caused spurious false positives (caught by a facade test failing after
   the change, before this note was written). Belief.status gained STUCK,
   triggered by a real gap-statistic bimodality test over force_required
   (not the imported version's interval-plateau heuristic, which is
   mathematically guaranteed to eventually fire on any heavily-tested,
   not-yet-confirmed belief regardless of whether it is actually blended).
   ConceptCodebook.split_concept now uses a seeded RNG (the imported version
   used the unseeded global numpy RNG, breaking this project's core
   determinism guarantee) and, given enough buffered recent observations,
   an SVD to find a genuine separating direction rather than a functionally
   negligible fixed perturbation.

   **Empirically validated, and the result is worth stating precisely.**
   For lever/switch and barrel/drum -- the catalogue's two OTHER look-alike
   pairs, whose true appearance separation is small but nonzero (~0.10, see
   D5/catalogue.py) -- splitting measurably improves future recognition:
   73/27 and 88/12 in testing, against 50/50 before. For crate/block, whose
   true separation is exactly zero, splitting still helps (each half
   accumulates independent evidence going forward) but future recognition
   stays at 50/50. That is not a bug in the split logic; it is the honest
   mathematical floor of what appearance alone can ever recover. No
   algorithm, however clever, can find a signal that genuinely is not
   present in the input. Where the real "hollow vs solid sphere" case sits
   on that spectrum -- whether it has SOME visual tell or none -- determines
   whether this mechanism can help it prospectively or only after the fact,
   per instance, through interaction. That is a fact about the physical
   scenario, not about the algorithm.

3. AI safety for irreversible actions. The imported gate only blocked
   selection once dominant_effect already showed TOPPLED/BROKE as the
   majority outcome -- reactive, requiring several things to have already
   broken. Replaced with Belief.irreversible_alpha/beta, a Beta posterior
   updated from every observed Outcome.irreversible (conditioned on
   success, mirroring remote_alpha/beta), starting at the uninformed
   Beta(1, 1) prior. A first attempt at a possibly-destructive verb is
   therefore moderately discounted by construction -- caution before the
   fact, not after -- and the discount relaxes quickly as evidence
   accumulates either way. select.py's scoring was also restructured from
   an ad hoc base + relevance + info_gain (which could make an
   already-settled, zero-score hypothesis positive again -- a real latent
   bug) to strictly multiplicative
   uncertainty/cost * safety * curiosity * goal, so nothing can revive a
   settled score.

**Self-improvement, requested separately mid-session.** There is no neural
network in the working core to gradient-train -- the affordance bank IS the
model (imagination/hypothesis.py's own docstring). scripts/run_lifelong.py
is the honest form this takes given that: run many episodes on one
persistent AffordanceMemory, derive goals from the agent's OWN confirmed
beliefs (never the oracle -- it is not cheating to pursue a goal about a door
once the agent has genuinely confirmed for itself that the door can open),
and report whether later episodes need less budget than earlier ones. Run
for real over 100 episodes: total interactions per 20-episode block fell
from 982 to 455, goal pursuit switched on automatically at episode 54 once
something was confirmed to reach OPENED, two real STUCK-to-split events
fired during ordinary play (not only the synthetic tests), and several
confirmed beliefs with a nonzero remote_rate turned up unprompted --
genuine secondary-affordance discovery. The interactions-per-newly-confirmed
figure is noisier block to block than the total; that is consistent with the
already-documented budget-grinding limitation and small per-block sample
size, and is reported as observed rather than smoothed over.

**On the real Isaac Lab evidence found in outputs/.** The committed
isaac_10ep_results.json / isaac_50ep_results.json are structured exactly
like this project's own IsaacWarehouseOracle.summary() and per-episode
verification-loop stats, which is reasonably strong evidence the Isaac Lab
backend (D6, written entirely unverified) was actually exercised on real
hardware and did not immediately fail. That data predates every fix in this
entry, so it validates that the backend runs at all, not the current code
specifically -- scripts/isaac_smoke_test.py and scripts/run_lama_isaac.py
remain the way to check the code as it stands now.

**A follow-up bug this entry's own fixes exposed, found by actually running
`scripts/run_lifelong.py` for 150 episodes rather than by unit tests alone.**
27 beliefs came back STUCK -- all resolving to crate/block, and all keyed on
already-retired (once-split) concepts. Because crate/block are appearance-
identical, splitting them can never succeed at separating future instances
(D7's own empirical validation already showed this: 50/50, unchanged from
chance); left uncapped, every fresh episode's crate/block instances
eventually re-accumulate enough evidence to trip STUCK again on whichever
descendant they land in, re-splitting it, cascading indefinitely. With the
same seed, this produced 73 concepts and 27 stuck beliefs where roughly a
dozen concepts were structurally expected.

Considered and rejected: gating the split on how strongly the recent
appearance evidence itself looks bimodal. Measured directly, it does not
discriminate -- a real trap's appearance-projection spread and a fake one's
came out at 0.078 vs 0.079 across 30 trials, because a small true gap
(lever/switch's ~0.10) does not detectably widen a noisy spread. There is no
cheap signal in `split_concept`'s own data to tell "this split will help"
from "this split cannot help" in advance.

Fixed instead with `Concept.generation` (0 for an ordinarily-merged concept,
N+1 for a concept produced by splitting a generation-N one) and
`AffordanceMemory.MAX_SPLIT_GENERATIONS = 1`: a lineage may be split once,
never again -- if it is still bimodal after that, the belief just stays
`STUCK`, which already stops budget going to that key without fragmenting
the concept space further. Re-running the same 150-episode, same-seed
scenario after the fix: 73 concepts down to 21, 27 stuck down to 2, and
CONFIRMED beliefs went UP (35 to 42) despite the same budget -- the
cascading re-splits were not just wasteful bookkeeping, they were actively
destroying accumulated evidence by resetting fresh, weak-prior beliefs on
every re-split. The validated lever/switch benefit (73/27, 88/12 sorting)
is unaffected, confirmed by rerunning that specific check after the change.

---

## D8. A real baseline comparison, and an honest negative-ish result on it

**Decision.** Built the empirical comparison this project had never actually
run: `verification/loop.py`'s `select_fn` seam lets any selection policy
drive the identical environment/memory/adjudication pipeline, so
`evaluation/baselines.py` (random, novelty-driven, uncertainty-only, and a
targeted safety-ablation) and the full system can be compared with nothing
else varying. `evaluate_precision_recall` now breaks recall out by affordance
role (`PRIMARY`/`SECONDARY`/`INCIDENTAL`), because an aggregate number cannot
show whether targeted verification helps specifically where the research
question says it should matter most -- see D5.

**Result, run for real at 15 seeds x 40 episodes (full account in
`docs/RESEARCH_FINDINGS.md`, raw data in
`outputs/baseline_comparison_with_ablation.json`).** LAMA's full system
clearly wins on overall recall,
primary-affordance recall, and interactions-per-confirmed-affordance (115.3
vs 184.5 for random, tightest variance of any policy). On secondary-
affordance recall specifically -- the actual headline metric -- it does
NOT win: the ordering inverts, uncertainty_only (0.122) > lama_no_safety
(0.113) > lama (0.103), each of LAMA's efficiency-oriented refinements
(cost-normalisation + curiosity, then safety) making secondary discovery
modestly worse while making everything else better.

**Why this was reported rather than reframed.** The pattern is
mechanistically explainable, not noise: secondary affordances in this
catalogue live disproportionately behind `PLACE_ON` (a specific relational
combination, not what a concept-level curiosity bonus rewards) and `TIP`
(irreversible by design for barrel/drum, which the safety discount exists
specifically to deprioritise). Both refinements do exactly what they were
built to do; they are simply in tension with fast discovery of the hardest,
rarest affordance class. Softening this into "LAMA wins overall" would have
been technically true and substantively misleading about the one number the
project's own design says matters most. The prior project in this
repository's history (`files/EmbodiedAI.zip`) is worth citing again here
specifically because its README modeled this: reporting that InfoNCE scores
below random initialisation, plainly, rather than omitting it.

**Not yet done.** `evaluate_retention`/`evaluate_transfer` are implemented
and tested for correctness but not run at this statistical scale. The two
concrete follow-ups this result motivates -- a relational-verb-specific
discovery bonus, and letting the planner relax the safety discount once it
has established no safer route reaches the same goal -- are proposed in
`docs/RESEARCH_FINDINGS.md`, not implemented.
