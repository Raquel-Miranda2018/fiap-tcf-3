# FASE 03 - Machine Learning Engineering - Deliverables

Base directory: /content/drive/MyDrive/FASE 03

Files included:
- model_output/: trained pipelines, model metadata, evaluation plots, shap, examples
- predict_pipeline_aligned.py : inference script that aligns produced features and predicts
- README.md (this file) and reports/analysis_report.md

How to run (Colab):
1. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Use the script to predict:

```bash
!python "/content/drive/MyDrive/FASE 03/predict_pipeline_aligned.py" --input "/content/drive/MyDrive/FASE 03/sample_to_predict.csv" --output "/content/drive/MyDrive/FASE 03/predictions.csv" --base_dir "/content/drive/MyDrive/FASE 03" --threshold 0.054292
```

Threshold:
- Default threshold used: from model_metadata or auto threshold file (see model_output/model_metadata_threshold_autof1.json)

