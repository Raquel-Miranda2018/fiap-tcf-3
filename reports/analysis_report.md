# Relatório de Análise — FIAP Tech Challenge (Fase 3)

## 1. Objetivo
Prever se um voo terá atraso superior a 15 minutos (classificação binária) usando dados de voos. Este relatório resume os passos principais, métricas do modelo e as features mais importantes encontradas.

---

## 2. Dados
- Origem: dataset de voos (arquivo bruto grande, não incluído no repositório por tamanho).  
- Caminhos locais usados durante a execução no Colab:
  - raw: `data/raw/flights.csv`
  - processed: `data/processed/voos_features.parquet` (quando disponível)
  - amostra (opcional, pequena): `data/sample/flights_sample.csv` (se existir)
- Observação: os dados brutos e os modelos treinados (`/models`) não foram adicionados ao repositório para evitar arquivos muito grandes. Indicações para reprodução abaixo.

---

## 3. Fluxo de trabalho executado
Notebooks executados (na ordem recomendada):
1. `notebooks/01_ingestao_limpeza_executado.ipynb` — ingestão e limpeza
2. `notebooks/02_eda_executado.ipynb` — análise exploratória
3. `notebooks/03_feature_engineering_executado.ipynb` — engenharia de features
4. `notebooks/04_modelo_supervisionado_executado.ipynb` — modelagem supervisionada
5. `notebooks/05_modelo_nao_supervisionado_executado.ipynb` — clustering/análise não supervisionada
6. `notebooks/06_resultados_finais_executado.ipynb` — resumo dos resultados

Artefatos gerados:
- Relatório: `reports/analysis_report.md` (este arquivo)
- Plots: `reports/plots_metrics.png` (gráficos de confusão e ROC)
- Métricas: `reports/metrics.csv`
- Modelos (salvos localmente/Colab): `/content/fiap-tcf-3/models/*.joblib` (não comitados)

---

## 4. Métricas principais (modelo final: Random Forest)
Resultados obtidos no conjunto de teste (subamostra utilizada: 200k no ambiente Colab):

- Accuracy (teste): 0.99
- ROC AUC (Random Forest): 0.9994
- Classification report (Random Forest):
  - Classe 0 (sem atraso >15 min): precision 1.00 — recall 0.99 — f1 1.00 — support 32971
  - Classe 1 (atraso >15 min): precision 0.96 — recall 1.00 — f1 0.98 — support 7029

Observação: A performance extremamente alta pode indicar forte sinal nas features escolhidas ou possível vazamento de informação; verificar validação temporal e origem das features é recomendado.

---

## 5. Top 10 features (importância — Random Forest)
As 10 features mais importantes (baseadas em feature importances do RF):

1. ATRASO_BINARIO — 0.4728  
2. ARRIVAL_DELAY — 0.1806  
3. LATE_AIRCRAFT_DELAY — 0.1236  
4. AIRLINE_DELAY — 0.0925  
5. AIR_SYSTEM_DELAY — 0.0336  
6. DEPARTURE_TIME — 0.0104  
7. SCHEDULED_DEPARTURE — 0.0081  
8. WEATHER_DELAY — 0.0080  
9. WHEELS_OFF — 0.0077  
10. TAXI_OUT — 0.0072

Interpretação rápida: várias variáveis de atraso explícitas aparecem entre as mais importantes — atenção a possíveis vazamentos (por exemplo, usar `ARRIVAL_DELAY` para prever `DEPARTURE_DELAY` pode ser circular dependendo da definição).

---

## 6. Como reproduzir (modo rápido)
1. Clonar o repo ou descompactar os arquivos no ambiente (Colab).
2. Instalar dependências:
   ```bash
   pip install -r requirements.txt
