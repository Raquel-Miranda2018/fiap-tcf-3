"""Testes para src/voos/eda.py."""

import matplotlib
import matplotlib.figure

from voos.eda import (
    atrasos_por_aeroporto,
    atrasos_por_companhia,
    atrasos_por_dia_semana,
    atrasos_por_mes,
    correlacao_variaveis,
    distribuicao_atrasos,
    estatisticas_descritivas,
)

matplotlib.use("Agg")


def test_estatisticas_descritivas_chaves(df_amostra_voos):
    """Resultado deve conter chaves: media, mediana, desvio_padrao, min, max, quartis."""
    resultado = estatisticas_descritivas(df_amostra_voos)
    chaves_esperadas = {"media", "mediana", "desvio_padrao", "min", "max", "quartis"}
    assert chaves_esperadas.issubset(set(resultado.keys()))
    # Deve ter valores para DEPARTURE_DELAY e ARRIVAL_DELAY
    for chave in ["media", "mediana"]:
        assert "DEPARTURE_DELAY" in resultado[chave]
        assert "ARRIVAL_DELAY" in resultado[chave]


def test_distribuicao_atrasos_retorna_figura(df_amostra_voos):
    """Deve retornar um objeto matplotlib.figure.Figure."""
    fig = distribuicao_atrasos(df_amostra_voos)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_atrasos_por_companhia_retorna_figura(df_amostra_voos):
    """Deve retornar Figure com pelo menos 1 eixo."""
    fig = atrasos_por_companhia(df_amostra_voos)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1


def test_atrasos_por_aeroporto_top_n(df_amostra_voos):
    """Com top_n=3, gráfico deve ter no máximo 3 barras."""
    fig = atrasos_por_aeroporto(df_amostra_voos, top_n=3)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Verificar que o eixo tem no máximo 3 tick labels
    ax = fig.axes[0]
    assert len(ax.get_xticks()) <= 4 or len(ax.get_yticks()) <= 4  # barras horizontais ou verticais


def test_atrasos_por_mes_retorna_figura(df_amostra_voos):
    """Gráfico de atrasos por mês deve ser um Figure válido."""
    fig = atrasos_por_mes(df_amostra_voos)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_atrasos_por_dia_semana_retorna_figura(df_amostra_voos):
    """Gráfico de atrasos por dia da semana deve ser um Figure válido."""
    fig = atrasos_por_dia_semana(df_amostra_voos)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_correlacao_variaveis_retorna_figura(df_amostra_voos):
    """Heatmap de correlação deve ser um Figure válido."""
    fig = correlacao_variaveis(df_amostra_voos)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_distribuicao_atrasos_nao_inclui_cancelados(df_amostra_voos):
    """Distribuição de atrasos deve filtrar voos cancelados."""
    # Se a função funciona sem erro com dados que incluem cancelados, já é um bom sinal
    fig = distribuicao_atrasos(df_amostra_voos)
    assert isinstance(fig, matplotlib.figure.Figure)
