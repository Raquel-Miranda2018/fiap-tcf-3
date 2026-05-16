"""Modelagem supervisionada — treino, avaliação e comparação de modelos."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

MODELOS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
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
        modelo_nome: 'logistic_regression', 'random_forest', ou 'gradient_boosting'.

    Returns:
        Modelo treinado.
    """
    if modelo_nome not in MODELOS:
        raise ValueError(f"Modelo '{modelo_nome}' não reconhecido. Opções: {list(MODELOS.keys())}")

    modelo = MODELOS[modelo_nome]()
    if params:
        modelo.set_params(**params)
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
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def comparar_modelos(resultados):
    """Cria tabela comparativa de modelos.

    Args:
        resultados: Lista de dicts retornados por avaliar_modelo (com chave 'modelo' adicionada).

    Returns:
        DataFrame com comparação lado a lado.
    """
    colunas = ["modelo", "accuracy", "precision", "recall", "f1", "roc_auc"]
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
    plt.close(fig)
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
    plt.close(fig)
    return fig
