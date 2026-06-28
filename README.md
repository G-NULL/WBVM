# Transport Research Experiments

This repository contains research notes, LaTeX sources, and experiment code for
weak bridge / RKHS-WBVM transport experiments.

## Project Layout

- `rkhs_wbvm_mnist_fid_experiments/`: MNIST image-generation experiments,
  remote runners, evaluation utilities, and tests.
- `rkhs_wbvm_synth_experiments/`: synthetic transport/RKHS-WBVM experiments.
- `paper_rewriting_output/` and `paper_2602_19600v1_src/`: manuscript notes and
  LaTeX sources.

Large datasets, model weights, generated PDFs, logs, and experiment outputs are
ignored by Git. Keep reproducible source code, scripts, configs, and concise
result summaries under version control; regenerate heavy artifacts when needed.

## Git Workflow

Use `main` as the stable branch and create short-lived feature branches for new
experiments or paper changes:

```bash
git switch -c feature/mnist-unified-logs
git status
git add <files>
git commit -m "feat: add MNIST training visualizations"
```

After creating a GitHub repository under `G-NULL`, connect it with:

```bash
git remote add origin https://github.com/G-NULL/<repo-name>.git
git push -u origin main
```
