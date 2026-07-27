# LAMA — Lifelong Affordance learning through counterfactual verificAtion

An embodied agent that learns what objects afford by **imagining** possible
interactions inside a world model, **verifying** only the hypotheses it is most
uncertain about through real interaction, and **keeping** only what verification
confirms.

## The research question

> Given a limited interaction budget, can an agent choose which of its imagined
> affordance hypotheses are worth testing in reality, and accumulate a knowledge
> base whose entries are both correct and durable?

The claim is about **interaction efficiency and knowledge retention**, not raw
task success. An agent that tries everything will eventually learn everything;
the question is whether targeted counterfactual verification gets there on a
fraction of the interactions, and whether what it stores stays right.

## The loop

```
observe
   |
   v
imagine        for each visible object, roll out every candidate action
   |           inside the world model -> predicted outcome + uncertainty
   v
select         rank hypotheses by epistemic uncertainty / expected information
   |           gain; pick the ones worth spending a real interaction on
   v
verify         execute the interaction; compare reality to the prediction
   |
   v
adjudicate     confirmed / refuted / inconclusive
   |
   v
remember       write only confirmed knowledge to the affordance bank, with
   |           provenance and confidence; revise on contradiction
   v
plan           downstream tasks consult the bank instead of re-deriving
```

The hard part is `select`. Uncertainty alone is not enough: some outcomes are
irreducibly random, and an agent that cannot separate "I do not know" from
"nothing could know" will burn its budget on noise. Making that distinction pay
off is the contribution.

## Status

Early. The environment is the current work.

| Component | State |
|---|---|
| Environment (hidden affordances, object-directed actions) | working |
| Perception encoder | not started |
| World model with calibrated uncertainty | not started |
| Counterfactual imagination | not started |
| Verification selection | not started |
| Verification execution and adjudication | not started |
| Lifelong affordance bank | not started |
| Planner over confirmed knowledge | not started |
| Evaluation harness and baselines | not started |

## The environment

A numpy warehouse of objects with hidden mass and hidden affordance profiles.
The agent sees geometry and a noisy appearance descriptor; it never sees an
object's kind, and object ids are opaque.

Twelve verbs: `approach, push, pull, lift, press, rotate, open, close, grasp,
release, place_on, tip`. Each costs budget, including when it fails.

Doors start locked, so they open only through a mechanism. Buttons, levers and
valves latch. The pressure plate does not: standing on it opens its door only
while you stand there, which is useless when you need to walk through. Putting
something heavy on it opens the door for good -- an affordance that cannot be
found by interacting with the plate alone.

Which brings in the trap. A `crate` and a `block` have **identical**
appearance descriptors. The block holds the plate down; the crate is too light.
Nothing the agent can see will separate them, and their one observable
difference is that the crate moves further when shoved. Two more pairs
(`lever`/`switch`, `barrel`/`drum`) sit at about 15% confusion, so appearance
is a useful prior there but not a reliable one.

About 2,600 interactions per second on one core, so 10^5 interactions takes
under a minute. That number is the reason this backend exists rather than a
physics one: the prior work in `files/EmbodiedAI.zip` produced no signal at 186
interaction records and concluded volume was the bottleneck.

## Evaluation plan

| Metric | What it shows |
|---|---|
| Interactions per confirmed affordance | Verification efficiency, the headline number |
| Precision of the affordance bank vs hidden ground truth | Whether "confirmed" means correct |
| Recall over the reachable affordance set | Whether targeting leaves blind spots |
| Retention across tasks and layouts | Whether the knowledge is lifelong or forgotten |
| Transfer to held-out object kinds | Whether the representation generalises |

Baselines to beat: random interaction, curiosity/novelty-driven interaction,
and uncertainty-driven interaction without the counterfactual and adjudication
steps. The third is the one that matters; the first two are sanity floors.

## Repository layout

```
lama/                  the LAMA system (new work)
docs/DECISIONS.md      why the project is built this way
aura_warehouse_sim/    prior work: AURA navigation-only warehouse prototype
AURA_Project_Plan.pdf  prior work: the AURA research plan this project grew from
files/EmbodiedAI.zip   prior work: an earlier object-function-discovery project
```

Prior work is retained deliberately. `files/EmbodiedAI.zip` in particular
contains a documented negative result — no unlabelled function-learning
objective beat random initialisation at 186 interaction records — that directly
motivates LAMA's emphasis on interaction efficiency.

## Setup

```bash
pip install -r requirements.txt
```

CPU is sufficient for everything currently planned.
