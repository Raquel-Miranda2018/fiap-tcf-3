"""Funções utilitárias compartilhadas — parse de horários HHMM, constantes."""

import math

# Colunas obrigatórias do dataset flights.csv
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


def parse_hhmm(valor):
    """Converte inteiro HHMM para tupla (horas, minutos).

    Retorna None se o valor for NaN ou inválido.
    """
    if valor is None:
        return None
    try:
        if math.isnan(valor):
            return None
    except (TypeError, ValueError):
        return None

    valor = int(valor)
    horas = valor // 100
    minutos = valor % 100
    return (horas, minutos)


def hhmm_para_minutos(valor):
    """Converte inteiro HHMM para minutos desde meia-noite.

    Retorna None se o valor for NaN.
    """
    resultado = parse_hhmm(valor)
    if resultado is None:
        return None
    horas, minutos = resultado
    return horas * 60 + minutos
