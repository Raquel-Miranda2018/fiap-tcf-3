"""Testes para src/voos/nao_supervisionado.py."""

import matplotlib
import numpy as np

from voos.nao_supervisionado import (
    executar_kmeans,
    executar_pca,
    interpretar_clusters,
    metodo_cotovelo,
    plotar_clusters,
    preparar_dados_clustering,
    silhouette_por_cluster,
)

matplotlib.use("Agg")


def _dados_clustering():
    """Cria dados numéricos simples para testes."""
    np.random.seed(42)
    return np.vstack([
        np.random.randn(10, 3) + [0, 0, 0],
        np.random.randn(10, 3) + [5, 5, 5],
        np.random.randn(10, 3) + [10, 0, 5],
    ])


def test_preparar_dados_clustering_escala():
    """Dados escalados devem ter média ~0 e desvio padrão ~1 por coluna."""
    dados = _dados_clustering()
    escalados = preparar_dados_clustering(dados)
    for i in range(escalados.shape[1]):
        assert abs(escalados[:, i].mean()) < 0.1
        assert abs(escalados[:, i].std() - 1.0) < 0.15


def test_executar_kmeans_numero_clusters():
    """Com n_clusters=3, resultado deve ter exatamente 3 labels distintos."""
    dados = preparar_dados_clustering(_dados_clustering())
    labels, _ = executar_kmeans(dados, n_clusters=3)
    assert len(set(labels)) == 3


def test_executar_kmeans_labels_tamanho():
    """Número de labels deve igualar número de amostras."""
    dados = preparar_dados_clustering(_dados_clustering())
    labels, _ = executar_kmeans(dados, n_clusters=3)
    assert len(labels) == len(dados)


def test_executar_pca_dimensoes():
    """Com n_components=2, resultado deve ter shape (n_amostras, 2)."""
    dados = _dados_clustering()
    dados_reduzidos, _ = executar_pca(dados, n_components=2)
    assert dados_reduzidos.shape == (30, 2)


def test_executar_pca_variancia_explicada():
    """Variância explicada acumulada deve estar entre 0 e 1."""
    dados = _dados_clustering()
    _, pca_model = executar_pca(dados, n_components=2)
    var_total = sum(pca_model.explained_variance_ratio_)
    assert 0 < var_total <= 1.0


def test_metodo_cotovelo_retorna_figura():
    """Gráfico do cotovelo deve retornar Figure válido."""
    dados = preparar_dados_clustering(_dados_clustering())
    fig = metodo_cotovelo(dados, k_range=range(2, 6))
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plotar_clusters_retorna_figura():
    """Scatter plot de clusters deve retornar Figure válido."""
    dados = preparar_dados_clustering(_dados_clustering())
    labels, _ = executar_kmeans(dados, n_clusters=3)
    dados_2d, _ = executar_pca(dados, n_components=2)
    fig = plotar_clusters(dados_2d, labels)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_interpretar_clusters_perfil():
    """Perfil de clusters deve ter uma linha por cluster."""
    import pandas as pd

    dados = _dados_clustering()
    dados_escalados = preparar_dados_clustering(dados)
    labels, _ = executar_kmeans(dados_escalados, n_clusters=3)
    df = pd.DataFrame(dados, columns=["a", "b", "c"])
    perfil = interpretar_clusters(df, labels)
    assert len(perfil) == 3


def test_silhouette_entre_menos1_e_1():
    """Silhouette score deve estar no intervalo [-1, 1]."""
    dados = preparar_dados_clustering(_dados_clustering())
    labels, _ = executar_kmeans(dados, n_clusters=3)
    score = silhouette_por_cluster(dados, labels)
    assert -1.0 <= score <= 1.0


def test_kmeans_reprodutibilidade():
    """Mesmos dados + mesmo random_state -> mesmos labels."""
    dados = preparar_dados_clustering(_dados_clustering())
    labels1, _ = executar_kmeans(dados, n_clusters=3, random_state=42)
    labels2, _ = executar_kmeans(dados, n_clusters=3, random_state=42)
    assert list(labels1) == list(labels2)
