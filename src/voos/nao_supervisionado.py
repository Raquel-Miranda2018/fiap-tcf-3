"""Modelagem não supervisionada — clustering (K-Means) e redução de dimensionalidade (PCA)."""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def preparar_dados_clustering(dados):
    """Escala os dados com StandardScaler (média 0, desvio padrão 1).

    Args:
        dados: ndarray ou DataFrame com features numéricas.

    Returns:
        ndarray escalado.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(dados)


def executar_kmeans(dados, n_clusters, random_state=42):
    """Executa K-Means clustering.

    Returns:
        Tupla (labels, modelo_kmeans).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(dados)
    return labels, kmeans


def executar_pca(dados, n_components=2):
    """Executa PCA para redução de dimensionalidade.

    Returns:
        Tupla (dados_reduzidos, modelo_pca).
    """
    pca = PCA(n_components=n_components)
    dados_reduzidos = pca.fit_transform(dados)
    return dados_reduzidos, pca


def metodo_cotovelo(dados, k_range=range(2, 11)):
    """Plota o gráfico do método do cotovelo para escolha do k ótimo.

    Returns:
        matplotlib.figure.Figure
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(dados)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(k_range), inertias, marker="o", linewidth=2, color="steelblue")
    ax.set_title("Método do Cotovelo — Escolha do k ótimo")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inércia (soma dos quadrados intra-cluster)")
    plt.tight_layout()
    return fig


def plotar_clusters(dados_2d, labels):
    """Plota scatter dos clusters em 2D (após PCA).

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        dados_2d[:, 0], dados_2d[:, 1],
        c=labels, cmap="viridis", alpha=0.7, edgecolors="black", s=60,
    )
    ax.set_title("Clusters — projeção PCA 2D")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    return fig


def interpretar_clusters(df, labels):
    """Cria perfil de cada cluster com média das features.

    Returns:
        DataFrame com uma linha por cluster.
    """
    df = df.copy()
    df["cluster"] = labels
    perfil = df.groupby("cluster").mean()
    return perfil


def silhouette_por_cluster(dados, labels):
    """Calcula o silhouette score global.

    Returns:
        Float entre -1 e 1.
    """
    return silhouette_score(dados, labels)
