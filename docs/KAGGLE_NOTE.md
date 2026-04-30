# Kaggle Runtime Note

Use GPU runtime on Kaggle for realistic D3QN convergence.

## Why curves may look negative in some packaged outputs

A lightweight numpy-only linear fallback agent can produce negative curves when PyTorch is unavailable. This does not change the report/plot/video packaging format.

## Recommended Kaggle run

```bash
pip install -r requirements.txt
python project_omega_m8.py --quick
```

For full training, remove `--quick` and allow long runtime with GPU enabled.
