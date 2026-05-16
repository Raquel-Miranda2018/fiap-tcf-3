# FIAP Tech Challenge — Fase 3: Análise de Atrasos de Voos

Projeto de Machine Learning Engineering para análise e predição de atrasos de voos nos EUA, utilizando o dataset público [Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays).

## Sobre o Projeto

Este projeto implementa um pipeline completo de ciência de dados:

1. **Exploração de Dados (EDA):** Estatísticas descritivas, visualizações de padrões temporais e espaciais, tratamento de valores ausentes.
2. **Modelagem Supervisionada:** Classificação binária (atraso >= 15 min), comparando Logistic Regression, Random Forest e Gradient Boosting.
3. **Modelagem Não Supervisionada:** K-Means clustering de aeroportos por perfil de atrasos, com visualização PCA.
4. **Apresentação Crítica:** Conclusões, limitações e propostas de melhorias.

## Setup

```bash
# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
make install

# Baixar o dataset e colocar em data/raw/
# flights.csv (~500MB) deve ser baixado manualmente do Kaggle
```

## Comandos

```bash
make test       # Executar testes
make test-cov   # Testes com cobertura
make lint       # Verificar estilo de código
make format     # Formatar código
make notebooks  # Executar todos os notebooks
make clean      # Limpar artefatos
```

## Estrutura do Projeto

```
├── src/voos/          # Módulos Python testáveis
│   ├── utils.py       # Parse HHMM, constantes
│   ├── ingestao.py    # Carregar CSV, validar schema
│   ├── limpeza.py     # Tratamento de valores ausentes
│   ├── eda.py         # Estatísticas e visualizações
│   ├── features.py    # Feature engineering
│   ├── supervisionado.py      # Modelos de classificação
│   └── nao_supervisionado.py  # Clustering e PCA
├── tests/             # Testes (pytest) — 55+ testes
├── notebooks/         # Jupyter notebooks (01 a 06)
├── data/raw/          # flights.csv (gitignored)
├── data/processed/    # Parquet processados (gitignored)
└── docs/              # Especificações e roteiro da apresentação
```

## Notebooks

| # | Notebook | Descrição |
|---|----------|-----------|
| 01 | Ingestão e Limpeza | Carregamento, validação de schema, tratamento de nulos |
| 02 | EDA | Estatísticas descritivas, visualizações, insights |
| 03 | Feature Engineering | Período do dia, estação, feriados, target binário |
| 04 | Modelo Supervisionado | Classificação com 3 algoritmos, métricas, ROC, feature importance |
| 05 | Modelo Não Supervisionado | K-Means em aeroportos, elbow method, PCA, perfil de clusters |
| 06 | Resultados Finais | Consolidação, limitações e próximos passos |

## Resultados

- **EDA:** Atrasos concentrados em meses de verão e dezembro. Sextas e domingos são os dias mais críticos.
- **Supervisionado:** Random Forest e Gradient Boosting superam a baseline e Logistic Regression. Horário de partida e aeroporto de origem são as features mais relevantes.
- **Não supervisionado:** 4 clusters de aeroportos identificados — hubs, regionais, problemáticos e eficientes.

## Equipe

- Bruno Bento

## Licença

MIT
