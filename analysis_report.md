# Relatório de Análise — Projeto FIAP TCF3

## Resumo rápido

- Tarefa: Classificação — prever atraso (atraso > 15 min)

- Dataset: voos_features.parquet


> Métricas não encontradas em reports/metrics.csv


## Top features (importância — Random Forest)

|                     |   importance |
|:--------------------|-------------:|
| ATRASO_BINARIO      |   0.472847   |
| ARRIVAL_DELAY       |   0.180645   |
| LATE_AIRCRAFT_DELAY |   0.123586   |
| AIRLINE_DELAY       |   0.0924743  |
| AIR_SYSTEM_DELAY    |   0.033551   |
| DEPARTURE_TIME      |   0.0104409  |
| SCHEDULED_DEPARTURE |   0.00810914 |
| WEATHER_DELAY       |   0.00797607 |
| WHEELS_OFF          |   0.00771554 |
| TAXI_OUT            |   0.00723508 |

## Observações e próximos passos

- Fazer validação temporal (se aplicável).
- Testar calibração de probabilidade e técnicas de sampling/SMOTE se houver desbalanceamento.
- Deploy: exportar pipeline e expos via API (FastAPI) ou salvar e servir com BentoML.
