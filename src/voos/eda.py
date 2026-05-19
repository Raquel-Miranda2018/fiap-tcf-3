"""Funções de análise exploratória — estatísticas descritivas e visualizações."""

import matplotlib.pyplot as plt
import seaborn as sns

COLUNAS_DELAY = ["DEPARTURE_DELAY", "ARRIVAL_DELAY"]


def estatisticas_descritivas(df):
    """Calcula estatísticas descritivas para colunas de atraso.

    Returns:
        Dicionário com chaves: media, mediana, desvio_padrao, min, max, quartis.
    """
    desc = df[COLUNAS_DELAY].describe()
    return {
        "media": df[COLUNAS_DELAY].mean().to_dict(),
        "mediana": df[COLUNAS_DELAY].median().to_dict(),
        "desvio_padrao": df[COLUNAS_DELAY].std().to_dict(),
        "min": df[COLUNAS_DELAY].min().to_dict(),
        "max": df[COLUNAS_DELAY].max().to_dict(),
        "quartis": {
            col: {
                "25%": desc.loc["25%", col],
                "50%": desc.loc["50%", col],
                "75%": desc.loc["75%", col],
            }
            for col in COLUNAS_DELAY
        },
    }


def distribuicao_atrasos(df):
    """Histograma da distribuição de atrasos (exclui voos cancelados).

    Returns:
        matplotlib.figure.Figure
    """
    df_valido = df[df["CANCELLED"] == 0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col in zip(axes, COLUNAS_DELAY):
        dados = df_valido[col].dropna()
        # Limitar o range para melhor visualização
        dados_clip = dados.clip(-60, 180)
        ax.hist(dados_clip, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribuição de {col}")
        ax.set_xlabel("Minutos")
        ax.set_ylabel("Frequência")
        ax.axvline(x=0, color="green", linestyle="--", alpha=0.7, label="Sem atraso")
        ax.axvline(x=15, color="red", linestyle="--", alpha=0.7, label="Limiar FAA (15 min)")
        ax.legend()

    plt.tight_layout()
    return fig


def atrasos_por_companhia(df):
    """Gráfico de barras: atraso médio por companhia aérea.

    Returns:
        matplotlib.figure.Figure
    """
    df_valido = df[df["CANCELLED"] == 0].copy()
    media = df_valido.groupby("AIRLINE")["DEPARTURE_DELAY"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    media.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Atraso médio na partida por companhia aérea")
    ax.set_xlabel("Companhia Aérea")
    ax.set_ylabel("Atraso médio (minutos)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def atrasos_por_aeroporto(df, top_n=20):
    """Gráfico de barras: aeroportos com maior atraso médio.

    Args:
        top_n: Número de aeroportos a exibir.

    Returns:
        matplotlib.figure.Figure
    """
    df_valido = df[df["CANCELLED"] == 0].copy()
    media = (
        df_valido.groupby("ORIGIN_AIRPORT")["DEPARTURE_DELAY"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    media.plot(kind="barh", ax=ax, color="coral", edgecolor="black")
    ax.set_title(f"Top {top_n} aeroportos com maior atraso médio na partida")
    ax.set_xlabel("Atraso médio (minutos)")
    ax.set_ylabel("Aeroporto de origem")
    plt.tight_layout()
    return fig


def atrasos_por_dia_semana(df):
    """Gráfico de atraso médio por dia da semana.

    Returns:
        matplotlib.figure.Figure
    """
    df_valido = df[df["CANCELLED"] == 0].copy()
    dias = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sáb", 7: "Dom"}
    media = df_valido.groupby("DAY_OF_WEEK")["DEPARTURE_DELAY"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [dias.get(d, str(d)) for d in media.index],
        media.values,
        color="mediumpurple",
        edgecolor="black",
    )
    ax.set_title("Atraso médio na partida por dia da semana")
    ax.set_xlabel("Dia da semana")
    ax.set_ylabel("Atraso médio (minutos)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    return fig


def atrasos_por_mes(df):
    """Gráfico de atraso médio por mês.

    Returns:
        matplotlib.figure.Figure
    """
    df_valido = df[df["CANCELLED"] == 0].copy()
    meses = {
        1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
    }
    media = df_valido.groupby("MONTH")["DEPARTURE_DELAY"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        [meses.get(m, str(m)) for m in media.index],
        media.values,
        marker="o",
        linewidth=2,
        color="teal",
    )
    ax.set_title("Atraso médio na partida por mês")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Atraso médio (minutos)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    return fig


def correlacao_variaveis(df):
    """Heatmap de correlação entre variáveis numéricas.

    Returns:
        matplotlib.figure.Figure
    """
    colunas_numericas = df.select_dtypes(include=["number"]).columns
    corr = df[colunas_numericas].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax, fmt=".2f")
    ax.set_title("Matriz de correlação — variáveis numéricas")
    plt.tight_layout()
    return fig
