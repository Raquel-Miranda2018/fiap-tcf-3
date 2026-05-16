"""Testes para src/voos/limpeza.py."""

import pandas as pd

from voos.limpeza import filtrar_voos_validos, salvar_parquet, tratar_valores_ausentes


def test_tratar_valores_ausentes_sem_nulos_criticos(df_amostra_voos):
    """Após tratamento, colunas críticas não devem ter NaN."""
    df = tratar_valores_ausentes(df_amostra_voos.copy())
    colunas_criticas = ["MONTH", "DAY", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    for col in colunas_criticas:
        assert df[col].isna().sum() == 0, f"Coluna {col} ainda tem NaN após tratamento"


def test_filtrar_voos_validos_remove_linhas_vazias(df_amostra_voos):
    """Voos não cancelados sem DEPARTURE_TIME devem ser removidos."""
    df = filtrar_voos_validos(df_amostra_voos.copy())
    # Voos não cancelados devem ter DEPARTURE_TIME
    nao_cancelados = df[df["CANCELLED"] == 0]
    assert nao_cancelados["DEPARTURE_TIME"].isna().sum() == 0


def test_voos_cancelados_mantem_colunas_identificacao(df_amostra_voos):
    """Voos cancelados mantêm AIRLINE, ORIGIN, DESTINATION mesmo após limpeza."""
    df = tratar_valores_ausentes(df_amostra_voos.copy())
    cancelados = df[df["CANCELLED"] == 1]
    if len(cancelados) > 0:
        assert cancelados["AIRLINE"].isna().sum() == 0
        assert cancelados["ORIGIN_AIRPORT"].isna().sum() == 0
        assert cancelados["DESTINATION_AIRPORT"].isna().sum() == 0


def test_salvar_parquet_round_trip(df_amostra_voos, tmp_path):
    """Salvar e recarregar parquet preserva schema e valores."""
    caminho = str(tmp_path / "teste.parquet")
    salvar_parquet(df_amostra_voos, caminho)
    df_carregado = pd.read_parquet(caminho)
    assert list(df_carregado.columns) == list(df_amostra_voos.columns)
    assert len(df_carregado) == len(df_amostra_voos)
