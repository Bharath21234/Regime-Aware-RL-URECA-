# Ideas to Strengthen the Paper for ICAIF '26 (research/experimentation only)

Deadline: 2026-08-02. This file tracks substantive research additions only,
no formatting/anonymization/page-limit items (those are tracked separately
and are mechanical, not scientific).

Status legend: [ ] not started · [~] in progress · [x] done

**Status snapshot 2026-07-04** (details in `results_log.md`):
- #1 training-mechanics fix: [x] done (GAE + batchfix); corrected single-period
  Hard-vs-Soft re-run + corrected 2-seed walkforward submitted (budget-limited,
  see log §6 caveat — first 2-seed walkforward ran old code and is superseded)
- #2 multi-seed walkforward: [~] 2-seed corrected re-run in the queue
- #3 PPO baseline: [x] all 3 variants × 3 seeds done (log §5); extra seeds 3-5
  script ready (`PPO/run_ppo_extraseeds.pbs`, needs budget)
- #4 transition-day analysis: [~] pipeline built+validated
  (`analysis_transition_days.py`); rerun on corrected walkforward when it lands
- #7 HMM state-count justification: [x] BIC/AIC sweep done
  (`figures/hmm_state_selection.py`) — K=4 optimal among non-degenerate models
- #8 gate interpretability: [x] figures done (`figures/make_gate_plot.py`)
- NEW: L2-confound control run script ready (`run_hard_l2match.pbs`, needs budget)
- NEW: K-regime ablation (K=2,3) parameterized end-to-end + script ready
  (`run_moe_kregimes.pbs`, needs budget)
- CFP confirmed: 8 pages ACM sigconf incl. references, double-blind, CMT,
  deadline 2026-08-02 AoE, notification 2026-09-27

---

## Tier 1 — Highest leverage, worth doing if at all possible

### 1. [~] Fix training instability (GAE + advantage normalization + LR annealing)
Implemented in all four `run.py` architectures (Hard/Baseline/MoE/Router).
Calibration queued, full rerun pending result.
**Why it matters**: directly targets the seed-instability problem that's shown
up everywhere in this project (Hard Routing's -13.88% seed collapse in Run B,
Router's 2/3 failed seeds). If this measurably tightens the Hard-vs-MoE gap
or improves absolute Sharpe, it's a stronger, more defensible headline result
than what you currently have.
**Next step**: run calibration, then full 3-seed rerun with `--reward_mode mv`.

### 2. [ ] Multi-seed walk-forward (3 seeds instead of 1)
**Why**: this is the most explicitly self-flagged limitation in your own
Discussion section ("seed-to-seed variance... is therefore unobserved at the
per-window level"). A reviewer is highly likely to flag this exact gap.
**Cost**: walkforward currently takes ~68h for 1 seed; 3 seeds means either
3x the wall time or running windows in parallel across multiple jobs.
**Prerequisite**: needs the GAE/normalization fixes ported into
`Walkforward/train.py` first if you want consistency with the main comparison
(currently only `run.py`'s 4 architectures have the fix).

### 3. [ ] Add a modern RL baseline (PPO)
**Why**: A2C is dated by 2026 standards; a reviewer working in current deep RL
literature may ask "why not PPO/SAC." `stable_baselines3` is already in
`requirements.txt`. Even a single PPO run on the Soft MoE architecture,
reported as "our architectural ranking holds when the underlying RL algorithm
changes from A2C to PPO," preempts this criticism cheaply.
**Cost**: moderate — needs a new training script, but the environment classes
are reusable as-is (gym-compatible).

### 4. [ ] Quantify the regime-transition story (not just one COVID anecdote)
**Why**: your strongest claim ("regime-aware methods add value during sharp
transitions") currently rests on one window (W1, COVID). Splitting each
walk-forward window's daily returns into "regime-stable days" vs
"regime-transition days" (using the HMM's own posterior — a day where the
most-likely state changes from the previous day) and reporting Sharpe
separately for each subset, across all methods, would turn this into a
quantified, paper-wide pattern instead of a single anecdote.
**Cost**: low — pure analysis on already-collected walk-forward output
(`window_*_rows.json` files), no retraining needed.

---

## Tier 2 — Good if time allows

### 5. [ ] Fold in the DSR ablation as a real result
DSR (Differential Sharpe Ratio reward) results already exist (job
3627948/3627949), but used the *old* training mechanics. If you do #1's full
rerun and it works well, redoing DSR with the same fixed mechanics gives a
clean second ablation axis (gating mechanism + reward shaping) essentially
for free since the infrastructure already exists.
**Open finding so far**: under the old mechanics, DSR numerically *reversed*
the architecture ranking (Hard > Router > MoE instead of MoE > Hard > Router)
but with n=3 this isn't statistically significant. Worth understanding
whether that reversal survives the new training mechanics before deciding
whether/how to mention it in the paper.

### 6. [ ] Sensitivity analysis on the risk-aversion coefficient (λ=0.5)
**Why**: shows the architectural ranking isn't fragile to one specific reward
hyperparameter choice. Cheap to run (just re-running with λ=0.25 and λ=1.0,
even at reduced epoch count) and a common reviewer ask for any paper with a
hand-picked hyperparameter in the core method.

### 7. [ ] Justify the 4-regime HMM choice
**Why**: currently 4 regimes is asserted, not justified. A quick BIC/AIC model
selection sweep (testing 2, 3, 4, 5, 6 states) showing 4 is a reasonable choice
preempts an easy reviewer question ("why 4?").
**Cost**: very low — this is a few lines of analysis on the already-fitted
HMM, not a retraining task.

### 8. [ ] Interpretability: visualize what the gate is actually doing
**Why**: a plot showing the HMM's regime probability vs. the Soft MoE's actual
expert-blending weights over time (especially during the COVID window) would
give visual, qualitative evidence for *why* the architecture works, not just
the aggregate statistics. Reviewers like seeing the mechanism, not just the
outcome.
**Cost**: low — pure analysis/plotting on already-trained models' saved
outputs (would need the model's intermediate routing weights logged, check if
already saved or easy to add to evaluation).

### 9. [ ] Get a cold mock-review pass from Prof Pun or a labmate
**Why**: given his recent ICAIF track record in this exact subfield, ask him
directly: "if you were reviewing this blind, what's the first thing you'd
flag?" Worth more than anything else on this list, costs nothing
computationally.

### 10. [ ] Anonymous code release
**Why**: increasingly expected in ML venues even without an explicit
checklist requirement. An anonymized GitHub/OSF link in the submission
(de-anonymized after acceptance) costs almost nothing since the code exists.

---

## Tier 2.5 — Added 2026-07-04 (second brainstorm)

### 15. [ ] Uniform-gate control run ("is it regime information, or just 4 heads?")
Train Soft RA-RL with the gate frozen at (1/4,1/4,1/4,1/4). Baseline controls
for *no* regime info, Router for *learned* routing; nothing yet controls for
pure multi-head capacity with no signal. If uniform-gate underperforms
real-gate Soft, the HMM's information content is proven to matter. One 3-seed
run, ~12h. (Blocked on budget.)

### 16. [ ] Save model checkpoints + test-period daily returns/weights
`torch.save` never called; test daily returns and weight histories not
persisted either — which blocks ensembling (#11), post-hoc gate-swap
ablations, Ledoit-Wolf tests on single-period runs, and transaction-cost
sensitivity (#17). Add to `run_experiment` in all variants before the next
batch of cluster runs so every future run produces reusable artifacts.

### 17. [ ] Turnover reporting + transaction-cost sensitivity
Cost penalty is 1bp — optimistic. Report average daily turnover per method
and net Sharpe at 5/10/25bp recomputed post-hoc from saved weight histories
(needs #16). Preempts the most common practitioner-reviewer complaint.

### 18. [ ] Cite recent ICAIF-venue literature
Related-work pass engaging 2-3 recent ICAIF papers (2023-25) on
regime-aware / RL trading. Reviewers are largely drawn from ICAIF authors;
absence of venue literature is conspicuous. Needs a literature search, no
compute.

### 19. [ ] Deflated Sharpe Ratio for the headline result
One computed number + one sentence (Bailey & López de Prado) showing the
headline Sharpe survives adjustment for multiple trials. Cheap,
finance-native rigor signal.

(Also done 2026-07-04, tracked in results_log.md: paired seed-matched
statistics replacing unpaired Welch [#20], Ledoit-Wolf/Jobson-Korkie-Memmel
Sharpe-difference tests on daily returns [#21], and the HMM-gated
Equal-Weight rule baseline [#22].)

### 23. [~] SAC variant suite (third algorithm)
Implemented 2026-07-04 in `Final/SAC/` (user decision, overriding the earlier
skip recommendation): full off-policy SAC mirror of the A2C/PPO suites —
same envs/rewards/trunk architectures for Baseline/Hard/Soft/Router, twin-Q,
auto-alpha, squashed-Gaussian wrapper (the one documented deviation: bounded
tanh actions, required for SAC well-posedness). 8/8 unit tests pass locally.
Next: `SAC/run_sac_calib.pbs` (2h, budget-permitting) before any full run.
Compute note: full 3-seed x 3-variant run is budget-blocked behind L2/K-reg/
PPO-seeds in the queue.

---

## Tier 3 — Lower priority / only if substantial time remains

### 11. [ ] Ensemble multiple seeds at inference
Average the action outputs of your 3 trained seeds into one ensemble policy
at test time, rather than treating them as independent draws. Often raises
realized Sharpe by cancelling idiosyncratic seed noise. **Blocked on**: model
checkpoints aren't currently saved to disk (`torch.save` never called), so
this needs that infrastructure added first.

### 12. [ ] Volatility targeting as a training-time input feature
(Not the post-hoc evaluation overlay we ruled out earlier — this would be an
actual input feature, e.g. trailing realized volatility, fed into the state
so the policy can learn its own position-sizing response to it.) More
invasive than the other items here, would need new environment/state changes
and full retraining to evaluate properly.

### 13. [ ] Richer non-RL benchmarks
Currently: Equal-Weight, Markowitz MVO, S&P 500 B&H. Adding risk parity or a
Black-Litterman benchmark would make the "EW beats RL" discussion more
rigorous, more benchmarks to contextualize against. Lower priority since the
existing three are already standard and sufficient, and ICAIF's 8-page limit
leaves little room for more tables.

### 14. [ ] Multiple-comparisons correction
You report 4 metrics × multiple architecture pairs. A sophisticated
statistical reviewer could ask whether you've corrected for multiple testing
(e.g., Holm-Bonferroni). Worth doing the correction and reporting whether your
significant results survive it, low cost, mostly just re-running existing
p-values through a correction formula.

---

## Open questions for you

1. How much cluster time are you actually willing to commit to in the
   remaining ~5 weeks, given the calibration + full rerun for #1 alone is
   already ~1.5 days? This determines how many of Tier 1/2 are realistic.
2. Has Prof Pun given any specific feedback yet that should reprioritize this
   list?
3. For #2 (multi-seed walkforward), do you want the GAE/normalization fixes
   ported into `Walkforward/train.py` first (for consistency with the main
   comparison), or run it with the old mechanics to save time?
4. Do you want me to start implementing any of these now, or hold until #1's
   calibration/full-run result comes back?
