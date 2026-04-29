"""
================================================================================
PROJECT OMEGA -- COLAB USAGE GUIDE
================================================================================
Author : Aryan Singh Chandel
Files  : project_omega_m1_m2.py  (Modules 1 & 2)
         project_omega_m3.py     (Module 3)
         project_omega_m4.py     (Module 4)
         project_omega_m5.py     (Module 5)
         project_omega_m6.py     (Module 6)
         project_omega_m7.py     (Module 7)
         project_omega_m8.py     (Module 8 + main orchestrator)
================================================================================

STEP-BY-STEP COLAB SETUP
=========================

CELL 1 -- Install dependencies
-------------------------------
!pip install -q gymnasium[box2d] torch fpdf2 reportlab python-pptx \
                wandb imageio moviepy scikit-learn seaborn scipy \
                pandas matplotlib numpy lxml Pillow

CELL 2 -- Upload files
-----------------------
from google.colab import files
# Upload all 8 project_omega_m*.py files when prompted
uploaded = files.upload()

CELL 3 -- Full overnight run (all 4 ablation variants)
-------------------------------------------------------
# This runs the COMPLETE pipeline:
#   - 4 variants x 2000 episodes each
#   - 100 eval episodes per variant
#   - All 8 plots (300 DPI)
#   - 11-page NeurIPS-style PDF
#   - 5-slide PPTX with speaker notes
#   - SHA-256 signed ZIP -> auto-synced to Google Drive

import sys
sys.argv = ["project_omega_m8.py"]   # no CLI flags = full overnight run
exec(open("project_omega_m8.py").read())

--------------------------------------------------------------------

CELL 3 (ALTERNATIVE) -- Quick smoke test (10 episodes, no ablation)
-------------------------------------------------------------------
import sys
sys.argv = ["project_omega_m8.py", "--quick"]
exec(open("project_omega_m8.py").read())

--------------------------------------------------------------------

CELL 3 (ALTERNATIVE) -- D3QN only, custom episode count
---------------------------------------------------------
import sys
sys.argv = ["project_omega_m8.py", "--skip-ablation", "--episodes", "500"]
exec(open("project_omega_m8.py").read())

--------------------------------------------------------------------

CELL 3 (ALTERNATIVE) -- Module 8 self-test only
-------------------------------------------------
import sys
sys.argv = ["project_omega_m8.py", "--test"]
exec(open("project_omega_m8.py").read())


OUTPUTS
=======
All artefacts are saved to:
    /content/drive/MyDrive/Project_Omega_D3QN/

Directory structure:
    models/
        checkpoints/
            best_d3qn.pth
            best_vanilla_dqn.pth
            best_dqn_per.pth
            best_dueling_dqn.pth
    plots/
        performance/
            performance_curve.png
            td_loss_error.png
            metrics_table.png
            final_metrics.csv
            training_metrics_*.csv
            all_variants_metrics.csv
            eval_summary.csv
        distributions/
            q_value_distribution.png
            latent_tsne.png
            state_visitation_heatmap.png
            action_confusion_matrix.png
        ablation/
            ablation_comparison.png
    reports/
        D3QN_Research_Paper.pdf   <- 11-page NeurIPS/ICML PDF
        Executive_Briefing.pptx   <- 5-slide C-suite deck
    video/
        best_agent_d3qn.gif       <- animated GIF of best episode
    logs/
        training_run.log
        tensorboard/
    output/
        Project_Omega_Release.zip <- SHA-256 signed artefact bundle
        manifest_hash.txt         <- integrity manifest


EXPECTED TRAINING TIME (Google Colab T4 GPU)
=============================================
Full ablation (4 variants x 2000 episodes):   6-10 hours
D3QN only     (1 variant  x 2000 episodes):   2-3 hours
Quick test    (1 variant  x 10   episodes):   < 5 minutes


W&B SETUP (optional)
====================
Before running, set your W&B username in RLConfig:
    cfg.wandb_entity = "your_username"

Or log in first:
    import wandb
    wandb.login()


SHA-256 VERIFICATION
====================
After the run, verify the ZIP integrity:

    import hashlib
    digest = hashlib.sha256(
        open("Project_Omega_Release.zip", "rb").read()
    ).hexdigest()

    # Compare with manifest_hash.txt
    manifest = open("manifest_hash.txt").read()
    print("PASS" if digest in manifest else "FAIL")
================================================================================
"""
