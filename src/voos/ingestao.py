"""Funções de ingestão de dados — carregar CSV, validar schema."""

import pandas as pd

from voos.utils import COLUNAS_FLIGHTS


def carregar_voos(caminho: str) -> pd.DataFrame:
    """Carrega o CSV de voos com tipos otimizados.

    Args:
        caminho: Caminho para o arquivo flights.csv.

    Returns:
        DataFrame com os dados carregados.
    """
    dtypes = {
        "YEAR": "int16",
        "MONTH": "int8",
        "DAY": "int8",
        "DAY_OF_WEEK": "int8",
        "AIRLINE": "category",
        "FLIGHT_NUMBER": "int32",
        "TAIL_NUMBER": "str",
        "ORIGIN_AIRPORT": "category",
        "DESTINATION_AIRPORT": "category",
        "DIVERTED": "int8",
        "CANCELLED": "int8",
        "CANCELLATION_REASON": "str",
    }
    df = pd.read_csv(caminho, dtype=dtypes, na_values=["", "NA"])
    return df


def validar_schema(df: pd.DataFrame) -> bool:
    """Verifica se o DataFrame contém todas as 31 colunas esperadas.

    Returns:
        True se todas as colunas estão presentes, False caso contrário.
    """
    colunas_presentes = set(df.columns)
    colunas_esperadas = set(COLUNAS_FLIGHTS)
    return colunas_esperadas.issubset(colunas_presentes)
