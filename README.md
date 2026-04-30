# Project Omega: Full Delivery Repository

![Python](https://img.shields.io/badge/Python-3.11-blue)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-D3QN-success)
![Status](https://img.shields.io/badge/Status-Delivery--Packaged-brightgreen)

This repository contains the full packaged Project Omega delivery and extracted artifacts.

## Included Delivery

- `delivery_full/source_code/` complete source modules
- `delivery_full/plots/` performance, distributions, ablation
- `delivery_full/video/` agent GIFs
- `delivery_full/reports/` research paper PDF + executive PPTX
- `delivery_full/models/checkpoints/` saved checkpoints
- `delivery_full/output/Project_Omega_Release.zip` packaged final artifact

## Visual Preview

![D3QN GIF](delivery_full/video/best_agent_d3qn.gif)

![Performance Curve](delivery_full/plots/performance/performance_curve.png)

![Ablation](delivery_full/plots/ablation/ablation_comparison.png)

## Kaggle Save + Version

1. Create a Kaggle Dataset and upload `Project_Omega_FULL_DELIVERY.zip`.
2. Create a Kaggle Notebook with GPU enabled and attach that Dataset.
3. Unzip in notebook:

```python
!unzip -q /kaggle/input/<your-dataset>/Project_Omega_FULL_DELIVERY.zip -d /kaggle/working/omega
```

4. Save notebook version via Kaggle UI using **Save Version**.

## Repository Layout

```text
.
+-- delivery_full/
+-- project_omega_m1_m2.py
+-- project_omega_m3.py
+-- project_omega_m4.py
+-- project_omega_m5.py
+-- project_omega_m6.py
+-- project_omega_m7.py
+-- project_omega_m8.py
+-- requirements.txt
```
