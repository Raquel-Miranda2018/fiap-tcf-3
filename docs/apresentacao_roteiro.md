# Roteiro da Apresentação — Tech Challenge Fase 3

**Duração total: 5-10 minutos**

## 1. Introdução (1 min)
- Apresentação da equipe
- Contexto do problema: impacto de atrasos de voos nos EUA
- Dataset utilizado: ~5.8M voos domésticos (2015), 31 colunas
- Objetivo: pipeline completo de ciência de dados com ML supervisionado e não supervisionado

## 2. EDA — Análise Exploratória (2 min)
- Visão geral dos dados: volume, distribuição de atrasos
- Padrões temporais: meses de verão e dezembro são os piores
- Aeroportos e companhias mais críticos
- Tratamento de dados: estratégia de limpeza e tratamento de nulos
- Mostrar 2-3 visualizações-chave

## 3. Feature Engineering (1 min)
- Features derivadas: período do dia, estação, feriados
- Target binário: limiar de 15 minutos (padrão FAA)
- Prevenção de data leakage: quais colunas foram removidas e por quê

## 4. Modelo Supervisionado (2 min)
- Modelos comparados: Logistic Regression, Random Forest, Gradient Boosting
- Tabela de métricas: accuracy, precision, recall, F1, ROC-AUC
- Curvas ROC e matrizes de confusão
- Feature importance: o que mais importa para prever atrasos
- Interpretação: o que os números significam em termos práticos

## 5. Modelo Não Supervisionado (2 min)
- Abordagem: K-Means em aeroportos agregados
- Método do cotovelo → k=4 clusters
- Perfil de cada cluster: hubs, regionais, problemáticos, eficientes
- Visualização PCA 2D
- Silhouette score

## 6. Conclusões e Próximos Passos (1-2 min)
- Principais achados
- Limitações: falta de dados meteorológicos, apenas 1 ano, desbalanceamento
- Melhorias propostas: dados de clima, XGBoost, validação temporal, dashboard
- Encerramento
