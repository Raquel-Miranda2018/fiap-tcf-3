Video script (5-7 min)
0) Opening (20s) — Introduce yourself and the challenge objective: predict flight delay (>15 min).
1) Data & EDA (60-90s) — describe dataset size, key features (datetime, origin/destination, airline, historical delays), mention handling of missing values and feature engineering (hour extraction, route stats).
2) Modeling (90-120s) — pipeline: feature_extract -> preproc -> model (LightGBM). Explain supervised task (classification) and unsupervised clustering brief.
3) Results (60-90s) — present AUC, best threshold (chosen by F1), precision/recall tradeoff, show ROC/PR and SHAP highlights (top 5 features).
4) Error analysis (60s) — FP/FN examples and likely causes (rare flights, data quality), what we'd improve.
5) Deployment (30-40s) — inference script, saving threshold, how to run in Colab or as batch job.
6) Closing (20s) — summary and next steps.

Notes for recording:
- Mention files included: model_output/, predict_pipeline_aligned.py, README.md, reports/analysis_report.md
- Keep slides simple: 6–8 slides mapping to sections above.
