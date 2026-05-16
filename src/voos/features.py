"""Feature engineering — criação de variáveis derivadas, encoding, seleção."""

import math

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Feriados federais dos EUA (mês, dia) — datas fixas
FERIADOS_EUA = [
    (1, 1),    # Ano Novo
    (7, 4),    # Dia da Independência
    (11, 11),  # Dia dos Veteranos
    (12, 25),  # Natal
    (12, 31),  # Véspera de Ano Novo
    (1, 2),    # Dia após Ano Novo (frequentemente feriado)
    (7, 3),    # Véspera da Independência
    (12, 24),  # Véspera de Natal
    (11, 26),  # Dia após Ação de Graças (aprox.)
    (11, 27),  # Ação de Graças (aprox.)
    (5, 25),   # Memorial Day (aprox.)
    (5, 26),   # Memorial Day (aprox.)
    (9, 1),    # Labor Day (aprox.)
    (9, 7),    # Labor Day (aprox.)
]


def criar_periodo_dia(hora_hhmm):
    """Classifica o horário HHMM em período do dia.

    Returns:
        'madrugada' (0-559), 'manha' (600-1159), 'tarde' (1200-1759), 'noite' (1800-2359)
    """
    if hora_hhmm is None:
        return None
    try:
        if math.isnan(hora_hhmm):
            return None
    except (TypeError, ValueError):
        pass

    hora_hhmm = int(hora_hhmm)
    if hora_hhmm < 600:
        return "madrugada"
    elif hora_hhmm < 1200:
        return "manha"
    elif hora_hhmm < 1800:
        return "tarde"
    else:
        return "noite"


def criar_estacao(mes):
    """Retorna a estação do ano (hemisfério norte — EUA).

    Returns:
        'inverno' (12,1,2), 'primavera' (3,4,5), 'verao' (6,7,8), 'outono' (9,10,11)
    """
    if mes in (12, 1, 2):
        return "inverno"
    elif mes in (3, 4, 5):
        return "primavera"
    elif mes in (6, 7, 8):
        return "verao"
    else:
        return "outono"


def criar_flag_feriado(mes, dia):
    """Indica se a data é próxima a um feriado federal dos EUA.

    Returns:
        True se feriado, False caso contrário.
    """
    return (mes, dia) in FERIADOS_EUA


def criar_target_atraso(delay_minutos, limiar=15):
    """Cria target binário: 1 se atrasado (>= limiar), 0 caso contrário.

    Args:
        delay_minutos: Atraso em minutos.
        limiar: Limiar em minutos (padrão FAA = 15).

    Returns:
        1, 0, ou None se delay for NaN.
    """
    if delay_minutos is None:
        return None
    try:
        if math.isnan(delay_minutos):
            return None
    except (TypeError, ValueError):
        pass

    return 1 if delay_minutos >= limiar else 0


def codificar_categoricas(df, colunas, metodo="label"):
    """Codifica colunas categóricas.

    Args:
        df: DataFrame.
        colunas: Lista de colunas a codificar.
        metodo: 'label' para LabelEncoder, 'onehot' para pd.get_dummies.

    Returns:
        DataFrame com colunas codificadas.
    """
    df = df.copy()
    if metodo == "label":
        for col in colunas:
            le = LabelEncoder()
            # Tratar NaN: converter para string temporariamente
            valores = df[col].astype(str).fillna("DESCONHECIDO")
            df[col] = le.fit_transform(valores)
    elif metodo == "onehot":
        df = pd.get_dummies(df, columns=colunas, drop_first=True)
    return df


def criar_features_completas(df):
    """Orquestra a criação de todas as features derivadas.

    Adiciona colunas: PERIODO_DIA, ESTACAO, FERIADO, ATRASO_BINARIO.
    """
    df = df.copy()
    df["PERIODO_DIA"] = df["SCHEDULED_DEPARTURE"].apply(criar_periodo_dia)
    df["ESTACAO"] = df["MONTH"].apply(criar_estacao)
    df["FERIADO"] = df.apply(lambda row: criar_flag_feriado(row["MONTH"], row["DAY"]), axis=1)
    df["ATRASO_BINARIO"] = df["DEPARTURE_DELAY"].apply(
        lambda x: criar_target_atraso(x, limiar=15)
    )
    return df


# Colunas que causam data leakage (informação do futuro)
COLUNAS_LEAKAGE = [
    "DEPARTURE_TIME", "DEPARTURE_DELAY", "ARRIVAL_TIME", "ARRIVAL_DELAY",
    "ELAPSED_TIME", "WHEELS_OFF", "WHEELS_ON", "TAXI_OUT", "TAXI_IN",
    "AIR_TIME", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY", "DIVERTED", "CANCELLED",
    "CANCELLATION_REASON", "ATRASO_BINARIO",
]


def selecionar_features_modelo(df):
    """Seleciona apenas features válidas para modelagem (sem leakage).

    Remove colunas que não estariam disponíveis no momento da previsão
    e colunas de identificação sem valor preditivo.
    """
    colunas_remover = set(COLUNAS_LEAKAGE) | {"YEAR", "FLIGHT_NUMBER", "TAIL_NUMBER"}
    colunas_manter = [col for col in df.columns if col not in colunas_remover]
    return df[colunas_manter]
