import joblib, os, sys, pandas as pd
MODEL = "/content/drive/MyDrive/FASE 03/model_output/pipeline_full.joblib"
if not os.path.exists(MODEL):
    raise FileNotFoundError("Pipeline não encontrado: " + MODEL)
pipe = joblib.load(MODEL)
if len(sys.argv) < 2:
    print("Uso: python predict_pipeline.py path_to_input.csv")
    sys.exit(1)
df = pd.read_csv(sys.argv[1], low_memory=False)
probs = pipe.predict_proba(df)[:,1]
df['pred_proba'] = probs
df['pred_label'] = (probs > 0.3495273889799213).astype(int)  # use threshold por F1 salvo
out = os.path.join(os.path.dirname(MODEL), "predictions_pipeline.csv")
df.to_csv(out, index=False)
print("Salvo:", out)
