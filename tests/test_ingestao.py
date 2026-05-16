"""Testes para src/voos/ingestao.py."""

from voos.ingestao import carregar_voos, validar_schema


def test_carregar_voos_colunas(caminho_csv_amostra):
    """CSV de amostra carregado deve ter 31 colunas."""
    df = carregar_voos(caminho_csv_amostra)
    assert len(df.columns) == 31


def test_carregar_voos_tipos(caminho_csv_amostra):
    """Colunas numéricas devem ser float/int, não object."""
    df = carregar_voos(caminho_csv_amostra)
    colunas_numericas = [
        "DEPARTURE_DELAY", "ARRIVAL_DELAY", "DISTANCE",
        "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME",
    ]
    for col in colunas_numericas:
        assert df[col].dtype in ("float64", "int64", "float32", "int32"), (
            f"Coluna {col} tem tipo {df[col].dtype}, esperado numérico"
        )


def test_validar_schema_valido(df_amostra_voos):
    """Schema válido retorna True."""
    assert validar_schema(df_amostra_voos) is True


def test_validar_schema_coluna_faltante(df_amostra_voos):
    """DataFrame sem uma coluna obrigatória retorna False."""
    df_sem_coluna = df_amostra_voos.drop(columns=["AIRLINE"])
    assert validar_schema(df_sem_coluna) is False
