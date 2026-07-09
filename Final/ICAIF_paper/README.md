# ICAIF '26 Submission — Working Directory

**Deadline: Aug 2, 2026 AoE** · 8 pages ACM sigconf **including references** ·
double-blind · submit via CMT · notification Sep 27 · in-person Milan
Nov 14-17 (>=1 author must attend, posters included; ask Prof Pun re
travel funding if accepted).

**Workshop fallback** (verified from icaif2026.org): accepted workshops
announced Aug 3-9; workshop paper deadlines TBD per workshop (likely late
Sep — may PRECEDE the Sep 27 main-track notification, so check each
workshop's dual-submission policy); workshop paper notification Oct 14;
workshops Nov 14-15. Candidate companion pieces if the same manuscript
can't be reused: multi-asset probe (results_log 7d) or the
mechanics-artifact story as a standalone reproducibility paper.

## How to compile

No LaTeX is installed on this Mac. Use **Overleaf** (recommended):

1. Zip this directory (`main.tex`, `references.bib`, plus a `figures/`
   subfolder — copy the needed PNGs/PDFs from `Final/figures/`).
2. New Overleaf project → Upload Project → select the zip.
3. Compiler: pdfLaTeX. acmart is preinstalled on Overleaf.

The class line is `\documentclass[sigconf,anonymous,review]{acmart}` —
`anonymous` suppresses author names, `review` adds line numbers. For
camera-ready, remove both options and restore the acknowledgments block.

## Pre-submission checklist

- [ ] **Resolve every `\pending{...}`** — grep for `pending`; the macro
      renders orange so stragglers are visible in the PDF. Main blockers:
  - [ ] Table 1 (corrected single-period A2C) — needs MoE seed 2 finisher
        (cluster job) + rerun of paired stats
  - [ ] Section 5.4 walk-forward — needs job wf_bf_2seed
        (`results_batchfix_2seed`), then `analysis_transition_days.py` and
        `analysis_paired_stats.py <wf_dir>` for JK-Memmel
  - [ ] Gated-EW table row: corrected Soft single-period Sharpe
  - [ ] Optional runs if budget allows: L2-match control, K=2/3 ablation,
        PPO extra seeds, SAC
- [ ] Copy figures from `Final/figures/` and uncomment the
      `\includegraphics` lines; **regenerate** walkforward_sharpe and the
      equity curve from the batchfix results (current versions plot
      superseded data)
- [ ] Fill `moedrlpm2025` authors and `istiaque2023hmm` venue in
      references.bib; ICAIF-venue citations pass (2–4 recent ICAIF papers,
      Improvement idea #18)
- [ ] Deflated-Sharpe sentence (Improvement idea #19)
- [ ] **Anonymization sweep**: no NTU/URECA/supervisor mentions; check PDF
      metadata (Overleaf: Menu → leave author blank); Istiaque/Pun/Yong
      cited in third person only; code link must be an anonymized repo
      (e.g. anonymous.4open.science) or "available on acceptance"
- [ ] Page budget: currently ~7 pp + refs estimated — after real tables and
      5 figures land it will exceed 8 pp; first cuts: Appendix per-seed
      tables (move to anonymized repo), architecture figure (merge 3
      diagrams into one row), trim Related Work

## Content decisions already baked in (vs the URECA paper)

- **Old-mechanics pooled Table I (n=9, incl. the p=0.0025 MaxDD claim) is
  NOT presented as a headline result** — it appears only inside §5.6 "The
  Mechanics Artifact" as the cautionary example. Headline single-period
  numbers are the corrected batchfix run.
- PPO framed as *mechanism validation* (trust region rescues hard
  switching), not as replication.
- Router collapse under both algorithms = the strongest claim; leads the
  results narrative.
- Gated-EW rule baseline = the "why RL" defense.
- Paired tests everywhere; Welch mentioned only as the weaker superseded
  choice.
