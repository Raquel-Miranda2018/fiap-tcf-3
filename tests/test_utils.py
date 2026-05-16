"""Testes para src/voos/utils.py."""

import math

import numpy as np

from voos.utils import hhmm_para_minutos, parse_hhmm


def test_placeholder():
    """Garante que o framework de testes funciona."""
    assert True


def test_parse_hhmm_meia_noite():
    """HHMM=0 deve retornar (0, 0)."""
    assert parse_hhmm(0) == (0, 0)


def test_parse_hhmm_valor_normal():
    """HHMM=1430 deve retornar (14, 30)."""
    assert parse_hhmm(1430) == (14, 30)


def test_parse_hhmm_limite():
    """HHMM=2359 deve retornar (23, 59)."""
    assert parse_hhmm(2359) == (23, 59)


def test_parse_hhmm_hora_baixa():
    """HHMM=100 deve retornar (1, 0) — 01:00."""
    assert parse_hhmm(100) == (1, 0)


def test_parse_hhmm_nan():
    """NaN deve retornar None."""
    assert parse_hhmm(np.nan) is None
    assert parse_hhmm(float("nan")) is None


def test_hhmm_para_minutos_valor_normal():
    """HHMM=1430 deve retornar 870 minutos (14*60 + 30)."""
    assert hhmm_para_minutos(1430) == 870


def test_hhmm_para_minutos_meia_noite():
    """HHMM=0 deve retornar 0 minutos."""
    assert hhmm_para_minutos(0) == 0


def test_hhmm_para_minutos_nan():
    """NaN deve retornar NaN."""
    resultado = hhmm_para_minutos(np.nan)
    assert resultado is None or (isinstance(resultado, float) and math.isnan(resultado))
