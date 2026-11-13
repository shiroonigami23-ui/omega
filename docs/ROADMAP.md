# 🗺 Roadmap — Project Omega

> Back to the [README](../README.md).

## Where the project stands

Status: **delivered**.

## Milestones

### v1.0 — Delivered

- [x] Modular 8-stage pipeline (setup → packaging)
- [x] 4-variant ablation with metrics CSVs
- [x] Auto-generated PDF paper, PPTX deck and agent GIFs
- [x] SHA-256 hashed release bundle

### v1.1 — Reach 'solved'

*Goal:* Get a stable 200+ rolling average on LunarLander.

- [ ] Extend training to 1,000–2,000 episodes per variant with learning-rate scheduling
- [ ] Hyper-parameter sweep (PER α/β, target-update τ, batch size) with Weights & Biases
- [ ] Report mean ± std over 3–5 random seeds instead of single runs

### v2.0 — Broader study

- [ ] Add Noisy-Nets, n-step returns and distributional C51 → full Rainbow comparison
- [ ] Run on more environments (CartPole, Acrobot, Atari Pong)
- [ ] Convert the exec()-chained modules into an installable package with a CLI (`omega train --variant d3qn`)

## 🔭 Long-term Vision

Grow Omega into a reusable "experiment-to-paper" framework: point it at any Gymnasium environment and any set of agent
variants, and get back a statistically sound comparison, a polished report and a reproducible release — in one command.

## 💡 Ideas Backlog (unscheduled)

- GitHub Action that re-renders plots from the CSVs on every push
- Interactive Streamlit dashboard for browsing runs
- Hugging Face model card + Space for the best agent

## How to help

Pick any unchecked item, open an issue to discuss the approach, then send a PR. Items in the earliest unfinished milestone are the highest priority.


---
*Last updated: 2026-11-13*
