"""Testes para src/voos/features.py."""

import numpy as np
import pandas as pd

from voos.features import (
    codificar_categoricas,
    criar_estacao,
    criar_features_completas,
    criar_flag_feriado,
    criar_periodo_dia,
    criar_target_atraso,
    selecionar_features_modelo,
)


def test_periodo_dia_madrugada():
    """HHMM entre 0 e 559 -> 'madrugada'."""
    assert criar_periodo_dia(0) == "madrugada"
    assert criar_periodo_dia(500) == "madrugada"


def test_periodo_dia_manha():
    """HHMM entre 600 e 1159 -> 'manha'."""
    assert criar_periodo_dia(600) == "manha"
    assert criar_periodo_dia(1100) == "manha"


def test_periodo_dia_tarde():
    """HHMM entre 1200 e 1759 -> 'tarde'."""
    assert criar_periodo_dia(1200) == "tarde"
    assert criar_periodo_dia(1700) == "tarde"


def test_periodo_dia_noite():
    """HHMM entre 1800 e 2359 -> 'noite'."""
    assert criar_periodo_dia(1800) == "noite"
    assert criar_periodo_dia(2359) == "noite"


def test_estacao_inverno():
    """Meses 12, 1, 2 -> 'inverno'."""
    assert criar_estacao(12) == "inverno"
    assert criar_estacao(1) == "inverno"
    assert criar_estacao(2) == "inverno"


def test_estacao_verao():
    """Meses 6, 7, 8 -> 'verao'."""
    assert criar_estacao(6) == "verao"
    assert criar_estacao(7) == "verao"
    assert criar_estacao(8) == "verao"


def test_estacao_primavera():
    """Meses 3, 4, 5 -> 'primavera'."""
    assert criar_estacao(3) == "primavera"


def test_estacao_outono():
    """Meses 9, 10, 11 -> 'outono'."""
    assert criar_estacao(9) == "outono"


def test_flag_feriado_natal():
    """25 de dezembro -> True."""
    assert criar_flag_feriado(12, 25) is True


def test_flag_feriado_dia_normal():
    """15 de março -> False."""
    assert criar_flag_feriado(3, 15) is False


def test_target_atraso_positivo():
    """Delay de 20 minutos com limiar 15 -> 1."""
    assert criar_target_atraso(20, limiar=15) == 1


def test_target_atraso_negativo():
    """Delay de 5 minutos com limiar 15 -> 0."""
    assert criar_target_atraso(5, limiar=15) == 0


def test_target_atraso_nan():
    """Delay NaN -> NaN."""
    resultado = criar_target_atraso(np.nan, limiar=15)
    assert resultado is None or (isinstance(resultado, float) and np.isnan(resultado))


def test_codificar_categoricas_label(df_amostra_voos):
    """Após label encoding, coluna deve ser numérica."""
    df = codificar_categoricas(df_amostra_voos.copy(), ["AIRLINE"], metodo="label")
    assert pd.api.types.is_numeric_dtype(df["AIRLINE"])


def test_criar_features_completas_novas_colunas(df_amostra_voos):
    """DataFrame resultante deve ter colunas: PERIODO_DIA, ESTACAO, FERIADO, ATRASO_BINARIO."""
    df = criar_features_completas(df_amostra_voos.copy())
    for col in ["PERIODO_DIA", "ESTACAO", "FERIADO", "ATRASO_BINARIO"]:
        assert col in df.columns, f"Coluna {col} não encontrada"


def test_selecionar_features_sem_vazamento(df_amostra_voos):
    """Features para modelo supervisionado NÃO devem incluir colunas com leakage."""
    df = criar_features_completas(df_amostra_voos.copy())
    df_features = selecionar_features_modelo(df)
    colunas_proibidas = [
        "DEPARTURE_TIME", "ARRIVAL_TIME", "ARRIVAL_DELAY", "ELAPSED_TIME",
        "WHEELS_OFF", "WHEELS_ON", "TAXI_OUT", "TAXI_IN",
        "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY", "AIR_TIME",
    ]
    for col in colunas_proibidas:
        assert col not in df_features.columns, f"Coluna de leakage {col} presente nas features"
