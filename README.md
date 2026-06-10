# Infraestrutura de Modelação Preditiva e Suporte à Decisão (COVID-19)
## Unidade Curricular: Sistemas de Business Intelligence — Semanas 13-15

Este repositório contém a solução modular de engenharia de dados e modelação estatística desenvolvida em Python. O objetivo do projeto é processar os dados históricos da pandemia, treinar algoritmos preditivos e extrair matrizes de dados e relatórios gráficos otimizados para consumo direto e visualização analítica.

---

## 1. Artefactos Criados e Estrutura de Ficheiros

O projeto foi estruturado de forma estritamente modular, garantindo o isolamento do código e prevenindo efeitos colaterais entre as diferentes questões de investigação.

### 1.1. Scripts de Extração e Modelação (`python/`)
- `process_q1.py`: Executa a normalização das séries temporais de incidência, realiza a divisão cronológica de dados e processa três abordagens algorítmicas concorrentes.
- `process_q2.py`: Implementa a engenharia de atributos (*feature engineering*) por desfasamento temporal e ajusta modelos de regressão estática para a mortalidade.
- `process_q3.py`: Aplica filtros de estabilização volumétrica sobre a taxa de letalidade para a remoção de ruído matemático inicial.
- `process_q4.py`: Atua como um guião de auditoria automatizada, extraindo as métricas de erro de todos os modelos e consolidando o balanço estatístico de validação.
- `gerar_graficos_relatorio.py`: Executa a leitura das matrizes de dados geradas e automatiza a exportação dos relatórios visuais e gráficos PNG para o documento final.

### 1.2. Datasets de Integração Criados (`datasets/`)
Estes ficheiros são o produto final do processamento em Python e constituem as fontes de dados diretas (Ground Truth) para o relatório final:
- `outputs_q1.csv`: Matriz temporal que emparelha a curva real de novos casos com as três curvas projetadas pelos modelos da Questão 1.
- `outputs_q2.csv`: Série temporal com os valores reais de mortalidade por milhão e as respetivas projeções de regressão da Questão 2.
- `outputs_q3.csv`: Matriz de dados emparelhada e sincronizada por data entre a taxa de letalidade de Portugal e a referência global após estabilização.
- `outputs_q4.csv`: Tabela estruturada de métricas de desempenho para alimentação de scorecards e painéis de controlo de qualidade de modelação.

---

## 2. Lógica do Código e Processo de Extração

Cada script processa o conjunto de dados em conformidade com as diretrizes do relatório crítico e do guião de e-Atividade.

### 2.1. Processamento e Previsão de Incidência (Questão 1)
O script `process_q1.py` isola a variável de novos casos diários transformando-a numa média móvel de 7 dias (`new_cases_7_day_avg_right`) para remover a variabilidade artificial provocada pelos atrasos de reporte de fim de semana.
- **Divisão Temporal:** Rejeita o baralhamento aleatório para evitar a fuga de informação (*data leakage*), dividindo a série de forma estritamente sequencial (80% treino para ajuste e 20% teste para validação).
- **Extração Algorítmica:** O código calcula simultaneamente três cenários preditivos sobre o conjunto de teste: uma linha de base pragmática (*Baseline Naïve* baseada no desfasamento de 1 dia), um modelo autorregressivo estatístico linear — ARIMA (1,1,1) — e um algoritmo de aprendizagem supervisionada não-linear (*Random Forest Regressor*) alimentado por atributos de atraso estruturais de 1, 7 e 14 dias.

### 2.2. Modelação de Risco e Engenharia de Atributos (Questão 2)
O script `process_q2.py` estabelece o modelo de monitorização do risco de mortalidade acumulada por milhão (`total_deaths_per_million`).
- **Controlo de Circularidade:** O código remove de forma total a Taxa de Letalidade (CFR) dos preditores independentes para mitigar o enviesamento decorrente de circularidade matemática.
- **Mecanismo de Desfasamento (*Lags*):** Para modelar estatisticamente o hiato temporal biológico que ocorre entre o contágio inicial e o desfecho de óbito, o script extrai e cria de forma dinâmica duas novas variáveis independentes: novos casos desfasados a 14 dias (`lag_14_new_cases`) e novos casos desfasados a 21 dias (`lag_21_new_cases`). O ajuste é validado através de modelos de Regressão Linear Múltipla e Regressão Polinomial de Grau 2.

### 2.3. Dinâmica de Estabilização da CFR (Questão 3)
O script `process_q3.py` analisa o comportamento dinâmico da Taxa de Letalidade (*Case Fatality Rate*).
- **Extração por Limiar:** O código aplica um filtro restritivo que descarta todos os registos cronológicos onde o volume acumulado de contágios seja inferior a 1.000 casos (`total_cases < 1000`). Este procedimento matemático elimina a volatilidade extrema típica do período inicial da epidemia. O guião extrai e sincroniza as curvas de Portugal e do mundo numa única matriz cronológica uniforme.

### 2.4. Consolidação do Balanço Estatístico (Questão 4)
O script `process_q4.py` automatiza a auditoria final da integridade da solução.
- **Extração de Métricas:** O código recolhe as previsões geradas nos outputs de Q1 e Q2, compara-as com os valores históricos observados no conjunto de teste e calcula os indicadores formais de validação: Erro Médio Absoluto (MAE), Raiz do Erro Quadrático Médio (RMSE) e o Coeficiente de Determinação (R2). O resultado é extraído numa tabela bidimensional compacta destinada ao desenho de tabelas de controlo de qualidade.

---

## 3. Instruções de Execução

Para reprocessar a infraestrutura e atualizar os ficheiros de dados e relatórios visuais, os guiões devem ser despoletados na linha de comandos de forma sequencial na pasta `python/`:

```bash
python process_q1.py
python process_q2.py
python process_q3.py
python process_q4.py
python gerar_graficos_relatorio.py