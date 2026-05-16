"""Testes para src/voos/supervisionado.py."""

import matplotlib

from voos.features import (
    codificar_categoricas,
    criar_features_completas,
    selecionar_features_modelo,
)
from voos.supervisionado import (
    avaliar_modelo,
    comparar_modelos,
    dividir_dados,
    importancia_features,
    plotar_curva_roc,
    plotar_matriz_confusao,
    treinar_modelo,
)

matplotlib.use("Agg")


def _preparar_dados(df_amostra_voos):
    """Helper: prepara dados de amostra para testes de modelagem."""
    df = criar_features_completas(df_amostra_voos.copy())
    # Filtrar apenas voos com target válido
    df = df[df["ATRASO_BINARIO"].notna()].copy()
    df["ATRASO_BINARIO"] = df["ATRASO_BINARIO"].astype(int)
    df_feat = selecionar_features_modelo(df)
    # Codificar categóricas
    cat_cols = df_feat.select_dtypes(include=["object", "category"]).columns.tolist()
    df_feat = codificar_categoricas(df_feat, cat_cols, metodo="label")
    return df_feat, df["ATRASO_BINARIO"]


def test_dividir_dados_proporcao(df_amostra_voos):
    """Com test_size=0.3, test set deve ter ~30% das linhas."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target, test_size=0.3)
    proporcao_teste = len(X_test) / (len(X_train) + len(X_test))
    assert 0.15 <= proporcao_teste <= 0.45  # margem para dados pequenos


def test_dividir_dados_sem_vazamento_target(df_amostra_voos):
    """Target column não deve estar em X_train ou X_test."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    assert "ATRASO_BINARIO" not in X_train.columns
    assert "ATRASO_BINARIO" not in X_test.columns


def test_treinar_modelo_retorna_objeto_com_predict(df_amostra_voos):
    """Modelo treinado deve ter método .predict()."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="random_forest")
    assert hasattr(modelo, "predict")
    assert hasattr(modelo, "predict_proba")


def test_avaliar_modelo_chaves(df_amostra_voos):
    """Resultado deve conter: accuracy, precision, recall, f1, roc_auc."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="random_forest")
    resultado = avaliar_modelo(modelo, X_test, y_test)
    chaves = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert chaves.issubset(set(resultado.keys()))


def test_avaliar_modelo_metricas_validas(df_amostra_voos):
    """Todas as métricas devem estar entre 0.0 e 1.0."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="random_forest")
    resultado = avaliar_modelo(modelo, X_test, y_test)
    for chave in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0.0 <= resultado[chave] <= 1.0, f"{chave} = {resultado[chave]} fora do range"


def test_comparar_modelos_formato(df_amostra_voos):
    """Tabela de comparação deve ter uma linha por modelo e colunas de métricas."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    resultados = []
    for nome in ["random_forest", "logistic_regression"]:
        modelo = treinar_modelo(X_train, y_train, modelo_nome=nome)
        res = avaliar_modelo(modelo, X_test, y_test)
        res["modelo"] = nome
        resultados.append(res)
    tabela = comparar_modelos(resultados)
    assert len(tabela) == 2
    assert "modelo" in tabela.columns


def test_importancia_features_soma(df_amostra_voos):
    """Importâncias de features (Random Forest) devem somar ~1.0."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="random_forest")
    imp = importancia_features(modelo, X_train.columns.tolist())
    assert abs(imp["importancia"].sum() - 1.0) < 0.01


def test_plotar_curva_roc_retorna_figura(df_amostra_voos):
    """Curva ROC deve retornar Figure válido."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="random_forest")
    y_prob = modelo.predict_proba(X_test)[:, 1]
    fig = plotar_curva_roc(y_test, y_prob, nome_modelo="RF")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plotar_matriz_confusao_retorna_figura(df_amostra_voos):
    """Matriz de confusão deve retornar Figure válido."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="random_forest")
    y_pred = modelo.predict(X_test)
    fig = plotar_matriz_confusao(y_test, y_pred)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_treinar_modelo_logistic_regression(df_amostra_voos):
    """Logistic Regression deve treinar sem erro."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, _, y_train, _ = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="logistic_regression")
    assert hasattr(modelo, "predict")


def test_treinar_modelo_random_forest(df_amostra_voos):
    """Random Forest deve treinar sem erro."""
    df_feat, target = _preparar_dados(df_amostra_voos)
    X_train, _, y_train, _ = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="random_forest")
    assert hasattr(modelo, "predict")
