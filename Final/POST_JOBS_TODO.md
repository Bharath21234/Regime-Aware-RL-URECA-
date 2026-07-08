# Post-Jobs To-Do — ICAIF '26 (deadline 2026-08-02 AoE)

Checklist for when the two running cluster jobs land. Companion to
`ICAIF_Improvement_Ideas.md` (what/why) and `results_log.md` (results).
Status: [ ] open · [x] done

---

## A. When the corrected single-period job (`URC_RL_mv_gae_bf`) finishes (~23h)

- [ ] A1. Pull results into FRESH local dirs — the job overwrites
      `results/hard` and `results/moe` on the cluster, and local `results/`
      root holds the old Run B loose seed files (don't clobber):
      ```bash
      scp -r 'bharath007@172.21.33.13:~/Regime-Aware-RL-URECA-/Final/results/hard' \
          ~/Documents/URECA/CODE/Regime-Aware-RL-URECA-/Final/results/hard_bf
      scp -r 'bharath007@172.21.33.13:~/Regime-Aware-RL-URECA-/Final/results/moe' \
          ~/Documents/URECA/CODE/Regime-Aware-RL-URECA-/Final/results/moe_bf
      scp 'bharath007@172.21.33.13:~/Regime-Aware-RL-URECA-/Final/URC_RL_mv_gae_bf.*' \
          ~/Documents/URECA/CODE/Regime-Aware-RL-URECA-/Final/
      ```
- [ ] A2. Analyze: per-seed table + paired t-tests (Soft vs Hard, corrected
      mechanics) → log as §8 in `results_log.md`
- [ ] A3. **THE DECISION** — does corrected-mechanics Soft still beat Hard at
      1000 epochs? Determines the ICAIF headline:
      - yes → corrected data replaces Tables I/II, "Soft > Hard" narrative
        (old pooled A/B/C becomes supporting/motivating evidence)
      - no/tie → reframe: "the Soft-vs-Hard gap is training-mechanics-
        dependent; the robust finding is external-signal vs learned routing"
      Everything downstream waits on this.
- [ ] A4. Cross-check consistency with PPO (§5) and pre-fix GAE (§4b) results

## B. When the corrected walk-forward (`wf_bf_2seed`) finishes (~36-45h)

- [ ] B0. If killed at the walltime wall mid-window: resubmit the same script,
      it resumes from cached windows in `results_batchfix_2seed/`
- [ ] B1. Pull `Walkforward/results_batchfix_2seed/` + `wf_bf_2seed.out/.err`
- [ ] B2. Aggregate + per-window analysis vs old Table III → log
- [ ] B3. Rerun the three waiting pipelines (one command each):
      ```bash
      python3 analysis_transition_days.py Walkforward/results_batchfix_2seed
      python3 analysis_paired_stats.py   Walkforward/results_batchfix_2seed
      # + Holm-Bonferroni pass over the final p-value set
      ```
- [ ] B4. Check W1 (COVID) and W6 specifically — did the old-code flips
      revert, or are they real?

## C. When budget frees (unused reservation returns on job completion) or top-up lands

- [ ] C0. FIRST: commit + push local changes (K/L2 parameterization,
      `--seed_start`, the 4 new PBS scripts, analysis scripts) and
      `git pull` on the cluster — the queued scripts depend on code not yet
      on the cluster
- [ ] C1. `qsub -l walltime=10:00:00 run_wf_soft_epochs_calib.pbs`
- [ ] C2. `qsub run_hard_l2match.pbs` (16h)
- [ ] C3. `qsub run_moe_kregimes.pbs` (30h)
- [ ] C4. `qsub PPO/run_ppo_extraseeds.pbs` (32h)
- [ ] C5. (still pending) email Prof Pun: budget top-up (~150-200 GPU-h ask)
      + mock-review request for mid-July

## D. Paper assembly

- [ ] D1. Update Tables I/II/III numbers; regenerate
      `figures/make_results_plots.py` + equity-curve figure with corrected data
- [ ] D2. Write ICAIF Results + Discussion around whichever narrative A3 selected
- [ ] D3. ICAIF skeleton: ACM sigconf 2-col, 8 pages incl. references,
      double-blind (can start before jobs finish)
- [ ] D4. Anonymization: byline, acknowledgments, "Following Istiaque, Pun,
      and Yong" phrasing (supervisor fingerprint), PDF metadata
- [ ] D5. ICAIF-venue citations pass (2-3 recent ICAIF regime/RL papers)
- [ ] D6. Deflated Sharpe Ratio sentence for the headline result (ideas #19)
- [ ] D7. Prof Pun mock-review draft by ~mid-July
- [ ] D8. Internal deadline **Jul 28**: CMT submission created, PDF compiles,
      anonymization verified. Hard deadline Aug 2 AoE.

---

Critical path: A1 → A2 → A3. Everything else hangs off that decision.
