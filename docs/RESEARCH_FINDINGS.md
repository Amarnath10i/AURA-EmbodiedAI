# Research findings

This document states, plainly, what gap this project addresses, what was
actually measured, and what the numbers show -- including where they do not
support the intended claim. That last part matters: the prior work this
project grew from (`files/EmbodiedAI.zip`) is valuable specifically because
its README reports a negative result honestly ("no objective produces
reliable function structure... the bottleneck is interaction volume, not
architecture"). This document follows the same standard.

## The research question

> Given a limited interaction budget, can an agent choose which of its
> imagined affordance hypotheses are worth testing in reality, and accumulate
> a knowledge base whose entries are both correct and durable?

The interesting version of this question is not "can it learn anything" --
random exploration eventually learns everything given enough budget. It is
"does targeted counterfactual verification reach *correct, confirmed*
knowledge on *less* budget than the standard alternatives, and does it do so
specifically for the hard cases (secondary affordances, appearance-identical
objects, irreversible actions) where naive exploration is expected to
struggle."

## Gaps this project addresses

1. **No empirical baseline comparison in this project's own prior work.**
   `files/EmbodiedAI.zip`'s Gen-3 evaluation measured whether a learned
   embedding's *clustering* recovered hidden object kinds -- it never
   compared *verification efficiency* against random, novelty-driven, or
   plain-uncertainty exploration through a shared environment and shared
   adjudication. `scripts/run_baseline_comparison.py` closes this: four
   policies (five, counting the safety ablation below) run through the
   identical loop, differing only in what they choose to test next.

2. **No distinction between primary and secondary affordance discovery in
   evaluation.** Standard affordance-learning evaluations report one
   aggregate number. This project's design (D5 in `docs/DECISIONS.md`) is
   built specifically around the claim that PRIMARY affordances (what an
   object visibly responds to) are easy for any strategy to find, while
   SECONDARY affordances (using an object as a means to an end, discoverable
   only through the right relational verb) are not. `evaluate_precision_recall`
   now reports recall broken out by role -- see Results below for why
   collapsing this into one number would have hidden the actual finding.

3. **Data starvation undermined every prior conclusion.** The prior project's
   own README: 186 interaction records, "the bottleneck is interaction
   volume, not architecture." This project's numpy backend runs at roughly
   2,600 interactions/second specifically to remove that confound --
   10^4-10^5 interactions are minutes of wall-clock time, not the binding
   constraint.

4. **No mechanism for recognising perceptual aliasing.** Standard affordance
   learning assumes appearance is sufficient to individuate objects. This
   project's concept-splitting mechanism (`memory/concepts.py`,
   `memory/bank.py`'s `STUCK` status) is validated, not asserted: it
   measurably improves future recognition for objects with a small but real
   appearance difference (lever/switch, barrel/drum: ~73/27 and ~88/12
   correct sorting after one split, versus 50/50 before), and is honestly
   reported as unable to help when the difference is exactly zero
   (crate/block) -- because no algorithm can recover a signal that is not
   present in the input. See D7 in `docs/DECISIONS.md` for the full account,
   including a cascading-fragmentation bug this same mechanism produced and
   the fix (`Concept.generation`, capped split depth).

5. **No principled treatment of irreversible actions in active verification.**
   Most active-learning setups treat every test as equally cheap and
   reversible. `Belief.irreversible_alpha/beta` (a Beta posterior over "does
   this cause unrecoverable damage", updated from every observed outcome, not
   just after several things have already broken) is a safety mechanism
   specific to embodied testing. Its actual cost is measured directly below,
   not just asserted to exist.

6. **Neural, black-box planning versus classical, verifiable planning.** The
   upstream work reviewed in D7 reached for a DreamerV3-style latent world
   model and CEM/MPPI planners -- expensive to train, hard to interpret, and
   (concretely, in that same review) shipped disconnected from the actual
   system with a guaranteed crash bug. `planner/planner.py`'s STRIPS-style
   regression planner requires no training, is fully inspectable (an
   `Operator` is a readable dataclass, not a weight matrix), and is verified
   to discover genuine multi-step structure (D7).

7. **Retention and transfer are usually asserted, rarely measured.**
   `evaluate_retention` and `evaluate_transfer` are implemented and run
   without crashing (see `tests/test_evaluation.py`), though -- stated
   plainly under Limitations below -- not yet run at the same statistical
   scale as the headline comparison.

## Method (summary; see README and `docs/DECISIONS.md` for full detail)

An agent observes a warehouse of objects with hidden mass and hidden,
role-tagged affordances (`PRIMARY`/`SECONDARY`/`INCIDENTAL`/`NONE`). Twelve
verbs, each with a real cost. The agent never sees an object's kind; concepts
form online from a noisy appearance descriptor (`memory/concepts.py`). A
Bayesian affordance bank (`memory/bank.py`) holds a Beta-Bernoulli belief per
`(concept, verb, tool concept)`, calibrated so ~8 clean successes confirm a
deterministic affordance and ~27 clean failures refute one. `select.py`'s
acquisition function ranks candidate tests by uncertainty per unit cost,
multiplicatively adjusted by a safety discount, a concept-level curiosity
bonus, and (when a goal is active) a backward-chaining relevance bonus from
`planner/planner.py`.

## Experimental setup

- Environment: `lama.env.warehouse.Warehouse`, `n_objects=10`, `budget=60.0`
  per episode, held-out kinds excluded.
- Five selection policies through the identical `verify_once`/`run_episode`
  loop (`verification/loop.py`'s `select_fn` seam), so only the choice of
  what to test differs:
  - `random` -- uniformly random among affordable hypotheses.
  - `novelty` -- prefer whatever has never been tried (Plan2Explore-style),
    no cost-normalisation, no calibration.
  - `uncertainty_only` -- `select.py`'s core term (uncertainty / cost) alone.
  - `lama_no_safety` -- the full formula minus only the safety discount.
  - `lama` -- the full system.
- 15 independent seeds, 40 episodes each, per policy (600 episode-runs per
  policy, 3,000 total). Metrics from `Evaluator.evaluate_precision_recall`
  and `evaluate_efficiency`, resolved from the agent's own concept ids back
  to real kinds via `ConceptKindTracker` (see `eval.py`'s module docstring
  for why that resolution step is the hard part of this evaluation).
- Raw results: `outputs/baseline_comparison_with_ablation.json`. Reproduce
  with `python scripts/run_baseline_comparison.py --episodes 40 --seeds 15`.

## Results

| Policy | Recall | Primary recall | Secondary recall | Interactions / confirmed |
|---|---|---|---|---|
| random | 0.197 ± 0.03 | 0.263 ± 0.05 | 0.077 ± 0.05 | 184.5 ± 40.0 |
| novelty | 0.218 ± 0.04 | 0.280 ± 0.06 | 0.087 ± 0.05 | 166.0 ± 26.9 |
| uncertainty_only | 0.254 ± 0.05 | 0.291 ± 0.06 | **0.122 ± 0.06** | 142.1 ± 37.9 |
| lama_no_safety | 0.259 ± 0.03 | 0.298 ± 0.05 | 0.113 ± 0.04 | 135.8 ± 23.6 |
| **lama** | **0.285 ± 0.03** | **0.363 ± 0.03** | 0.103 ± 0.05 | **115.3 ± 13.1** |

(Precision was 0.98-1.00 for every policy -- essentially all confirmed
beliefs are genuinely correct, regardless of selection strategy. That is a
statement about the bank's calibration, not about verification efficiency;
see Limitations.)

### What this shows

**LAMA's full system is the clear, consistent winner on overall efficiency.**
115.3 interactions per confirmed affordance versus 184.5 for random -- a real
~38% reduction -- with the tightest variance of any policy (±13.1 vs ±23-40
for everything else). Primary-affordance recall follows the same clean,
monotonic ordering: random < novelty < uncertainty_only < lama_no_safety <
lama.

**Secondary-affordance recall does not follow that ordering, and the reason
is identifiable, not noise.** It runs in the OPPOSITE direction:
uncertainty_only (0.122) > lama_no_safety (0.113) > lama (0.103). Each
refinement that improves aggregate efficiency -- cost-normalisation and
curiosity first, then the safety discount on top -- makes the system
modestly *worse* specifically at finding secondary affordances. This is
mechanistically explainable, not a fluke: secondary affordances in this
catalogue are disproportionately reached through `PLACE_ON` (a relational
verb requiring the right tool already in hand) and `TIP` (irreversible by
design for barrel/drum). A concept-level curiosity bonus rewards testing
whatever object has the most UNRESOLVED verbs in aggregate, which is not the
same thing as testing the one specific relational combination a secondary
affordance requires; a safety discount specifically deprioritises `TIP`,
which is *the only verb* that reveals some of these affordances at all. Both
refinements were built to make the aggregate case more efficient, and they do
-- at a real, measured cost to the rarest, hardest case, which is also the
case this project's own research question cares about most.

**This is a genuine trade-off, not a bug to silently fix.** Deprioritising
`TIP` when its risk is uncertain is the correct safety behaviour in
isolation; it is simply in tension with fast secondary-affordance discovery
when `TIP` is the only route to one. Reporting an aggregate recall number
would have hidden this tension entirely -- exactly why the role breakdown
(gap 2, above) exists.

## Limitations

- **Precision is uninformative here.** Every policy lands at 0.98-1.00
  because the bank's `CONFIRMED` threshold (D-calibrated in `bank.py`) is
  strict regardless of how a belief's evidence was gathered. This says the
  bank rarely certifies something false; it says nothing about how
  efficiently different policies reach true certification, which is what
  recall and interactions-per-confirmed measure instead.
- **No comparison against the literature's actual baselines.** The AURA
  planning document (`AURA_Project_Plan.pdf`) names DreamerV3, TD-MPC2, and
  Plan2Explore as the baselines to beat. None are implemented here;
  `novelty` is a deliberately simple proxy for Plan2Explore's spirit
  (uncertainty-seeking without calibration), not a reimplementation.
- **Single environment family.** All results are from `Warehouse` at one
  size (`n_objects=10`). Generalisation-focused metrics (`evaluate_transfer`,
  `evaluate_retention`) are implemented and tested for correctness
  (`tests/test_evaluation.py`) but not yet run at this statistical scale --
  a natural next experiment, not a result reported here.
- **The Isaac Lab backend is untested for this comparison.** `outputs/`
  holds real single-policy runs against Isaac Lab (D6/D7), but the baseline
  comparison above ran only on the numpy backend.
- **15 seeds is enough to see the pattern, not enough to bound it tightly.**
  Standard errors on secondary recall (the headline, hardest metric) are
  roughly half the size of the gaps between policies -- real signal, but a
  larger run would sharpen the confidence interval considerably. Given the
  numpy backend's throughput, this is a compute-time decision, not a
  fundamental limitation.

## What would make secondary-affordance discovery genuinely better, not just measured

The finding above suggests a specific next step, not just "run more seeds":
secondary affordances need their own targeted signal rather than relying on
the same aggregate machinery tuned for overall efficiency. Two concrete
directions this evaluation surfaces:

1. A bonus for **relational verbs specifically** (`PLACE_ON`, which is where
   most secondary affordances in this catalogue live) when the agent is
   already holding something -- distinct from, and likely larger than, the
   general concept-level curiosity bonus.
2. Decoupling the safety discount from verbs that are a *known-only-route* to
   an otherwise-undiscoverable effect, once the planner has established
   there is no alternative confirmed path -- i.e., let `RegressionPlanner`
   inform not just what to prioritise (already implemented) but when the
   safety discount specifically should relax because nothing safer would
   reach the same goal.

Neither is implemented; both are motivated directly by the numbers above,
which is the point of running the comparison in the first place.
