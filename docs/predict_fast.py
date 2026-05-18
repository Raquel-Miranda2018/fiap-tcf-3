import joblib, os, pandas as pd, numpy as np, json, sys

MODEL_DIR = "/content/drive/MyDrive/FASE 03/model_output"
MODEL_FILE = os.path.join(MODEL_DIR, "lgb_model_final.joblib")
META_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Modelo nao encontrado: " + MODEL_FILE)
model = joblib.load(MODEL_FILE)

# carregar metadata
if os.path.exists(META_FILE):
    with open(META_FILE, "r") as f:
        meta = json.load(f)
    features = meta.get("features_columns", None)
    best_threshold = float(meta.get("best_threshold", 0.2873167362008228))
else:
    features = None
    best_threshold = 0.2873167362008228

if len(sys.argv) < 2:
    print("Uso: python predict_fast.py path_to_input.csv")
    sys.exit(1)
input_csv = sys.argv[1]
df = pd.read_csv(input_csv, low_memory=False)

# extrair horas se existirem
def extract_hour_col(df, col):
    if col in df.columns:
        colnum = pd.to_numeric(df[col], errors='coerce')
        df[col + "_hour"] = (colnum // 100).fillna(0).astype(int)
        df.drop(columns=[col], inplace=True)

for c in ['SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL','DEPARTURE_TIME','ARRIVAL_TIME']:
    extract_hour_col(df, c)

# preencher NAs simples
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
cat_cols = [c for c in df.columns if c not in num_cols]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# alinhar features de forma eficiente
if features is not None:
    # cria DataFrame com colunas esperadas e valores default 0 (mais rápido que inserir coluna a coluna)
    X = pd.DataFrame(0, index=df.index, columns=features)
    # preenche intersecção entre df e features
    common = [c for c in features if c in df.columns]
    if len(common) > 0:
        X.loc[:, common] = df.loc[:, common].values
else:
    X = df.select_dtypes(include=[np.number])

probs = model.predict_proba(X)[:,1]
labels = (probs > best_threshold).astype(int)

df_out = df.copy()
df_out['pred_proba'] = probs
df_out['pred_label'] = labels
out_path = os.path.join(MODEL_DIR, "predictions_from_model_fast.csv")
df_out.to_csv(out_path, index=False)
echo "Predictions saved to:" $out_path
