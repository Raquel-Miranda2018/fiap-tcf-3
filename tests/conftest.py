"""Fixtures compartilhadas para todos os testes do projeto."""

import numpy as np
import pandas as pd
import pytest

# Colunas do dataset flights.csv conforme dicionário de dados
COLUNAS_FLIGHTS = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER",
    "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY",
    "TAXI_OUT", "WHEELS_OFF", "SCHEDULED_TIME", "ELAPSED_TIME",
    "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN",
    "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY",
    "DIVERTED", "CANCELLED", "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]


@pytest.fixture
def df_amostra_voos():
    """DataFrame de amostra com 30 linhas cobrindo todos os edge cases.

    Inclui:
    - 20 voos normais com valores realistas
    - 3 voos cancelados (CANCELLED=1, delays NaN, CANCELLATION_REASON preenchido)
    - 2 voos desviados (DIVERTED=1)
    - 2 voos com partida à meia-noite (HHMM=0)
    - 1 voo com atraso muito longo (300 min)
    - 1 voo com atraso negativo (chegou adiantado)
    - 1 voo com DEPARTURE_TIME NaN mas não cancelado (dados ausentes)
    """
    np.random.seed(42)
    n = 30

    companhias = ["AA", "UA", "DL", "WN", "US"]
    aeroportos = ["ATL", "ORD", "LAX", "DFW", "JFK", "SFO", "SEA", "BOS"]

    dados = {
        "YEAR": [2015] * n,
        "MONTH": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  1, 3, 5, 7, 9, 11, 2, 6, 8, 10, 12, 4, 6, 1, 7, 11, 3, 5],
        "DAY": list(range(1, 29)) + [30, 31],
        "DAY_OF_WEEK": [((i % 7) + 1) for i in range(n)],
        "AIRLINE": [companhias[i % len(companhias)] for i in range(n)],
        "FLIGHT_NUMBER": list(range(100, 100 + n)),
        "TAIL_NUMBER": [f"N{1000 + i}AA" for i in range(n)],
        "ORIGIN_AIRPORT": [aeroportos[i % len(aeroportos)] for i in range(n)],
        "DESTINATION_AIRPORT": [aeroportos[(i + 3) % len(aeroportos)] for i in range(n)],
        "SCHEDULED_DEPARTURE": [
            0, 600, 800, 1200, 1430, 1800, 2100, 2359,  # horários variados
            0, 730, 915, 1045, 1330, 1545, 1715, 1900,
            2200, 500, 1100, 1400, 1630, 2000, 100, 300,
            845, 1015, 1230, 1500, 1745, 2030,
        ],
        "DEPARTURE_TIME": [
            5, 610, 815, 1225, 1430, 1830, 2115, 2359,
            10, 745, 920, 1050, 1340, 1600, 1720, 1910,
            2215, 510, 1105, 1415, 1645, 2010, 110, 315,
            np.nan, np.nan, np.nan,  # 3 cancelados (indices 24, 25, 26)
            1505,
            np.nan,  # indice 28: não cancelado mas sem dados
            2040,
        ],
        "DEPARTURE_DELAY": [
            5, 10, 15, 25, 0, 30, 15, 0,
            10, 15, 5, 5, 10, 15, 5, 10,
            15, 10, 5, 15, 15, 10, 10, 15,
            np.nan, np.nan, np.nan,  # cancelados
            5,
            np.nan,  # sem dados
            10,
        ],
        "TAXI_OUT": np.random.randint(5, 30, n).astype(float).tolist(),
        "WHEELS_OFF": [
            20, 625, 830, 1240, 1445, 1845, 2130, 15,
            25, 800, 935, 1105, 1355, 1615, 1735, 1925,
            2230, 525, 1120, 1430, 1700, 2025, 125, 330,
            np.nan, np.nan, np.nan, 1520, np.nan, 2055,
        ],
        "SCHEDULED_TIME": np.random.randint(60, 360, n).astype(float).tolist(),
        "ELAPSED_TIME": np.random.randint(55, 370, n).astype(float).tolist(),
        "AIR_TIME": np.random.randint(45, 340, n).astype(float).tolist(),
        "DISTANCE": np.random.randint(200, 3000, n).astype(float).tolist(),
        "WHEELS_ON": [
            300, 900, 1100, 1500, 1700, 2100, 2350, 200,
            310, 1030, 1200, 1330, 1600, 1830, 1950, 2130,
            100, 800, 1350, 1630, 1900, 2250, 400, 600,
            np.nan, np.nan, np.nan, 1730, np.nan, 2300,
        ],
        "TAXI_IN": np.random.randint(3, 20, n).astype(float).tolist(),
        "SCHEDULED_ARRIVAL": [
            300, 900, 1100, 1500, 1700, 2100, 2350, 200,
            300, 1030, 1200, 1330, 1600, 1830, 1950, 2130,
            100, 800, 1350, 1630, 1900, 2250, 400, 600,
            1200, 1400, 1600, 1730, 1900, 2300,
        ],
        "ARRIVAL_TIME": [
            310, 910, 1110, 1520, 1650, 2115, 2355, 155,
            320, 1045, 1205, 1335, 1610, 1845, 1955, 2140,
            110, 805, 1355, 1640, 1910, 2300, 410, 555,
            np.nan, np.nan, np.nan, 1740, np.nan, 2310,
        ],
        "ARRIVAL_DELAY": [
            10, 10, 10, 20, -50, 15, 5, -5,  # indice 4: adiantado; indice 7: adiantado
            20, 15, 5, 5, 10, 15, 5, 10,
            10, 5, 5, 10, 10, 10, 10, -5,
            np.nan, np.nan, np.nan,  # cancelados
            300,  # indice 27: atraso muito longo
            np.nan,  # sem dados
            10,
        ],
        "DIVERTED": [0] * 20 + [1, 1] + [0] * 8,  # indices 20, 21 desviados
        "CANCELLED": [0] * 24 + [1, 1, 1] + [0] * 3,  # indices 24, 25, 26 cancelados
        "CANCELLATION_REASON": [np.nan] * 24 + ["A", "B", "C"] + [np.nan] * 3,
        "AIR_SYSTEM_DELAY": [0.0] * 24 + [np.nan] * 3 + [0.0, np.nan, 0.0],
        "SECURITY_DELAY": [0.0] * 24 + [np.nan] * 3 + [0.0, np.nan, 0.0],
        "AIRLINE_DELAY": [0.0] * 24 + [np.nan] * 3 + [0.0, np.nan, 0.0],
        "LATE_AIRCRAFT_DELAY": [0.0] * 24 + [np.nan] * 3 + [0.0, np.nan, 0.0],
        "WEATHER_DELAY": [0.0] * 24 + [np.nan] * 3 + [0.0, np.nan, 0.0],
    }

    df = pd.DataFrame(dados)
    assert list(df.columns) == COLUNAS_FLIGHTS, "Schema da fixture não bate com dicionário de dados"
    assert len(df) == 30, "Fixture deve ter exatamente 30 linhas"
    return df


@pytest.fixture
def caminho_csv_amostra(df_amostra_voos, tmp_path):
    """Salva o DataFrame de amostra como CSV temporário e retorna o caminho."""
    caminho = tmp_path / "flights_amostra.csv"
    df_amostra_voos.to_csv(caminho, index=False)
    return str(caminho)
