
# Relatório Técnico - Predição de Atrasos de Voos (Tech Challenge Fase 3)

## 1. Metodologia
- Algoritmo: LightGBM (Gradient Boosting)
- Otimização: Optuna para busca de hiperparâmetros.
- Métrica Principal: AUC-ROC de 0.9314, indicando excelente capacidade de separação entre classes.

## 2. Principais Descobertas (Insights SHAP)
As variáveis que mais impactam o atraso são:
1. Hora Real de Partida: O efeito cascata ao longo do dia é o principal fator.
2. Médias Históricas: Aeroportos e companhias com histórico de atraso tendem a repetir o padrão.
3. Distância: Voos de longa distância apresentam comportamentos de atraso diferentes de voos curtos regionais.

## 3. Gestão de Erros (FP/FN)
- Falsos Negativos (Atrasou e o modelo não previu): Ocorrem geralmente em voos que tinham tudo para ser pontuais (manhã, boas companhias), sugerindo que fatores externos não mapeados (clima extremo repentino, problemas mecânicos) são a causa.
- Falsos Positivos (O modelo previu atraso, mas foi pontual): Ocorrem em situações de risco alto onde a operação conseguiu recuperar o tempo.

## 4. Próximos Passos Sugeridos
- Integração de dados meteorológicos em tempo real via API.
- Clusterização de aeroportos por volume de tráfego para melhorar a generalização do modelo.
