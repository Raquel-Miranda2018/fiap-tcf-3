"""Testes de integração — pipeline completo e validação de entregáveis."""

import os

from voos.features import (
    codificar_categoricas,
    criar_features_completas,
    selecionar_features_modelo,
)
from voos.ingestao import carregar_voos, validar_schema
from voos.limpeza import filtrar_voos_validos, tratar_valores_ausentes
from voos.supervisionado import avaliar_modelo, dividir_dados, treinar_modelo

PROJETO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pipeline_completo_amostra(caminho_csv_amostra):
    """Pipeline inteiro: ingestão -> limpeza -> features -> modelo -> avaliação sem erro."""
    # Ingestão
    df = carregar_voos(caminho_csv_amostra)
    assert validar_schema(df)

    # Limpeza
    df = tratar_valores_ausentes(df)
    df = filtrar_voos_validos(df)
    assert len(df) > 0

    # Features
    df = criar_features_completas(df)
    assert "ATRASO_BINARIO" in df.columns

    # Preparar para modelo
    df_valido = df[df["ATRASO_BINARIO"].notna()].copy()
    df_valido["ATRASO_BINARIO"] = df_valido["ATRASO_BINARIO"].astype(int)
    target = df_valido["ATRASO_BINARIO"]
    df_feat = selecionar_features_modelo(df_valido)
    cat_cols = df_feat.select_dtypes(include=["object", "category"]).columns.tolist()
    df_feat = codificar_categoricas(df_feat, cat_cols, metodo="label")

    # Modelo
    X_train, X_test, y_train, y_test = dividir_dados(df_feat, target)
    modelo = treinar_modelo(X_train, y_train, modelo_nome="random_forest")
    resultado = avaliar_modelo(modelo, X_test, y_test)
    assert "accuracy" in resultado
    assert 0.0 <= resultado["accuracy"] <= 1.0


def test_todos_notebooks_existem():
    """Verifica que os 6 notebooks existem em notebooks/."""
    notebooks_dir = os.path.join(PROJETO_ROOT, "notebooks")
    esperados = [
        "01_ingestao_limpeza.ipynb",
        "02_eda.ipynb",
        "03_feature_engineering.ipynb",
        "04_modelo_supervisionado.ipynb",
        "05_modelo_nao_supervisionado.ipynb",
        "06_resultados_finais.ipynb",
    ]
    for nb in esperados:
        assert os.path.exists(os.path.join(notebooks_dir, nb)), f"Notebook {nb} não encontrado"


def test_readme_contem_secoes_obrigatorias():
    """README.md deve conter seções obrigatórias."""
    readme_path = os.path.join(PROJETO_ROOT, "README.md")
    with open(readme_path) as f:
        conteudo = f.read()
    for secao in ["## Setup", "## Comandos", "## Estrutura do Projeto", "## Equipe"]:
        assert secao in conteudo, f"Seção '{secao}' não encontrada no README"


def test_modulos_importaveis():
    """Todos os módulos do pacote voos devem ser importáveis."""
    from voos import (
        eda,  # noqa: F401
        features,  # noqa: F401
        ingestao,  # noqa: F401
        limpeza,  # noqa: F401
        nao_supervisionado,  # noqa: F401
        supervisionado,  # noqa: F401
        utils,  # noqa: F401
    )
