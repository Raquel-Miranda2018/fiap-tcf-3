"""Funções de limpeza de dados — tratamento de valores ausentes, filtros."""

import pandas as pd


def tratar_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Trata valores ausentes de acordo com a natureza de cada coluna.

    Estratégia:
    - Colunas de identificação (AIRLINE, aeroportos, datas): não devem ter NaN.
    - Colunas de tempo real (DEPARTURE_TIME, ARRIVAL_TIME): NaN é esperado para
      voos cancelados — mantemos como está.
    - Colunas de delay breakdown: NaN quando não há atraso — preenchemos com 0.
    """
    # Colunas de delay breakdown: NaN significa sem atraso, preencher com 0
    colunas_delay_breakdown = [
        "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
    ]
    for col in colunas_delay_breakdown:
        if col in df.columns:
            # Só preencher com 0 para voos não cancelados e com atraso >= 0
            mascara = (df["CANCELLED"] == 0) & df[col].isna()
            df.loc[mascara, col] = 0.0

    return df


def filtrar_voos_validos(df: pd.DataFrame) -> pd.DataFrame:
    """Remove voos inválidos — não cancelados sem dados de partida.

    Voos cancelados (CANCELLED=1) são mantidos mesmo sem DEPARTURE_TIME.
    Voos não cancelados sem DEPARTURE_TIME são dados corrompidos e são removidos.
    """
    # Manter cancelados + não cancelados que têm DEPARTURE_TIME
    mascara = (df["CANCELLED"] == 1) | df["DEPARTURE_TIME"].notna()
    return df[mascara].reset_index(drop=True)


def salvar_parquet(df: pd.DataFrame, caminho: str) -> None:
    """Salva DataFrame em formato Parquet."""
    df.to_parquet(caminho, index=False)
