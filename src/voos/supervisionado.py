"""Modelagem supervisionada — treino, avaliação e comparação de modelos."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

MODELOS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=100, random_state=42
    ),
    "random_forest_balanced": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
    "gradient_boosting_balanced": lambda: HistGradientBoostingClassifier(
        max_iter=100, random_state=42
    ),
}


def dividir_dados(X, y, test_size=0.2, random_state=42):
    """Divide dados em treino e teste com estratificação.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def treinar_modelo(X_train, y_train, modelo_nome="random_forest", **params):
    """Treina um modelo de classificação.

    Args:
        modelo_nome: chave do dicionário MODELOS.

    Returns:
        Modelo treinado.
    """
    if modelo_nome not in MODELOS:
        raise ValueError(f"Modelo '{modelo_nome}' não reconhecido. Opções: {list(MODELOS.keys())}")

    modelo = MODELOS[modelo_nome]()
    if params:
        modelo.set_params(**params)

    # HistGradientBoosting não suporta class_weight; usar sample_weight
    if modelo_nome == "gradient_boosting_balanced":
        sw = compute_sample_weight("balanced", y_train)
        modelo.fit(X_train, y_train, sample_weight=sw)
    else:
        modelo.fit(X_train, y_train)
    return modelo


def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo com métricas de classificação.

    Returns:
        Dict com accuracy, precision, recall, f1, roc_auc, confusion_matrix.
    """
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def comparar_modelos(resultados):
    """Cria tabela comparativa de modelos.

    Args:
        resultados: Lista de dicts retornados por avaliar_modelo (com chave 'modelo' adicionada).

    Returns:
        DataFrame com comparação lado a lado.
    """
    colunas = ["modelo", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    linhas = []
    for res in resultados:
        linhas.append({col: res.get(col) for col in colunas})
    return pd.DataFrame(linhas)


def importancia_features(modelo, feature_names):
    """Retorna DataFrame com importância de cada feature.

    Funciona com modelos que possuem atributo feature_importances_ (Random Forest, etc.).

    Returns:
        DataFrame com colunas: feature, importancia. None se não suportado.
    """
    if not hasattr(modelo, "feature_importances_"):
        return None
    importancias = modelo.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importancia": importancias,
    }).sort_values("importancia", ascending=False)
    return df


def plotar_curva_roc(y_test, y_prob, nome_modelo="Modelo"):
    """Plota a curva ROC.

    Returns:
        matplotlib.figure.Figure
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"{nome_modelo} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Baseline (AUC = 0.5)")
    ax.set_xlabel("Taxa de Falso Positivo")
    ax.set_ylabel("Taxa de Verdadeiro Positivo")
    ax.set_title("Curva ROC")
    ax.legend()
    plt.tight_layout()
    return fig


def plotar_matriz_confusao(y_test, y_pred):
    """Plota a matriz de confusão.

    Returns:
        matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No horário", "Atrasado"],
        yticklabels=["No horário", "Atrasado"],
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    plt.tight_layout()
    return fig


def plotar_curva_precision_recall(y_test, y_prob, nome_modelo="Modelo"):
    """Plota a curva Precision-Recall.

    Returns:
        matplotlib.figure.Figure
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = np.mean(y_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals,
            label=f"{nome_modelo} (AP = {ap:.3f})", linewidth=2)
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalência = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    plt.tight_layout()
    return fig


def analise_threshold(y_test, y_prob, nome_modelo="Modelo"):
    """Analisa precision, recall e F1 para diferentes thresholds de decisão.

    Returns:
        (fig, df): Figure matplotlib e DataFrame com métricas por threshold.
    """
    thresholds = np.arange(0.10, 0.91, 0.05)
    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_test, y_pred_t, zero_division=0),
            "recall": recall_score(y_test, y_pred_t, zero_division=0),
            "f1": f1_score(y_test, y_pred_t, zero_division=0),
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["threshold"], df["precision"], "b-o", label="Precision", markersize=4)
    ax.plot(df["threshold"], df["recall"], "r-o", label="Recall", markersize=4)
    ax.plot(df["threshold"], df["f1"], "g-o", label="F1 Score", markersize=4)
    melhor_f1_idx = df["f1"].idxmax()
    melhor_t = df.loc[melhor_f1_idx, "threshold"]
    ax.axvline(x=melhor_t, color="gray", linestyle=":", alpha=0.7,
               label=f"Melhor F1 (threshold={melhor_t:.2f})")
    ax.set_xlabel("Threshold de Decisão")
    ax.set_ylabel("Métrica")
    ax.set_title(f"Precision / Recall / F1 por Threshold — {nome_modelo}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, df
