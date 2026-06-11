# Infraestrutura de Modelação Preditiva e Suporte à Decisão — COVID-19

## Unidade Curricular

Sistemas de Business Intelligence
e-Atividade — Semanas 13 a 15
Mestrado em Engenharia Informática e Tecnologia Web

## Enquadramento

Este repositório contém a infraestrutura analítica desenvolvida para a fase final do trabalho experimental da unidade curricular, centrada na aplicação de modelos preditivos a dados epidemiológicos da COVID-19.

A solução dá continuidade às fases anteriores do projeto, nas quais foram estudadas arquiteturas Big Data, processos de engenharia de dados, limpeza, pré-tratamento e definição de questões de investigação. Nesta etapa, o foco desloca-se para a modelação, validação estatística, geração de outputs analíticos e suporte à decisão.

A implementação foi desenvolvida em Python, utilizando bibliotecas de análise de dados, aprendizagem automática, séries temporais e visualização gráfica.

## Objetivo do projeto

O objetivo principal é avaliar de que forma modelos preditivos podem apoiar a análise da evolução epidemiológica da COVID-19, com particular incidência no caso português.

A solução procura responder a quatro questões analíticas:

1. Prever a incidência diária suavizada em Portugal através da comparação entre Baseline Naïve, ARIMA e Random Forest.
2. Avaliar se variáveis de incidência desfasadas no tempo permitem explicar a evolução da mortalidade acumulada por milhão de habitantes.
3. Analisar a estabilização da Taxa de Letalidade (CFR) em Portugal face à referência mundial.
4. Consolidar e comparar as métricas de desempenho dos modelos implementados.

## Estrutura do repositório

```text
e-Atividade--3TE-SEMANA5-7/
│
├── datasets/
│   ├── covid_global_limpo.csv
│   ├── outputs_q1.csv
│   ├── outputs_q2.csv
│   ├── outputs_q3.csv
│   └── outputs_q4.csv
│
├── graficos/
│   ├── grafico_q1_incidencia.png
│   ├── grafico_q2_mortalidade.png
│   ├── grafico_q3_cfr.png
│   └── grafico_q4_metricas.png
│
├── python/
│   ├── process_q1.py
│   ├── process_q2.py
│   ├── process_q3.py
│   ├── process_q4.py
│   └── gerar_graficos_relatorio.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Nota sobre datasets grandes

O ficheiro `covid_paises_limpo.csv` não deve ser incluído diretamente no GitHub quando excede os limites recomendados de versionamento.

Para executar os scripts localmente, o ficheiro deve existir na pasta:

```text
datasets/covid_paises_limpo.csv
```

Este ficheiro resulta da fase anterior de extração, limpeza e pré-tratamento dos dados COVID-19.

## Dependências

As bibliotecas necessárias encontram-se definidas no ficheiro `requirements.txt`:

```text
pandas
numpy
matplotlib
scikit-learn
statsmodels
```

Para instalar as dependências, pode ser utilizado o comando:

```bash
pip install -r requirements.txt
```

## Ordem de execução dos scripts

Os scripts devem ser executados a partir da pasta `python/`, pela seguinte ordem:

```bash
python process_q1.py
python process_q2.py
python process_q3.py
python process_q4.py
python gerar_graficos_relatorio.py
```

A ordem é relevante porque o script `process_q4.py` depende dos outputs gerados pelas Questões 1 e 2, enquanto o script de gráficos depende dos outputs consolidados.

## Questão 1 — Incidência diária suavizada

O script `process_q1.py` modela a incidência diária suavizada em Portugal, utilizando a variável:

```text
new_cases_7_day_avg_right
```

São comparados três modelos:

1. Baseline Naïve, com previsão baseada no valor do dia anterior.
2. ARIMA (1,1,1), enquanto modelo estatístico clássico de séries temporais.
3. Random Forest Regressor, usando variáveis de atraso temporal a 1, 7 e 14 dias.

O output gerado é:

```text
datasets/outputs_q1.csv
```

Este ficheiro contém a série real e as previsões dos três modelos.

## Questão 2 — Mortalidade acumulada por milhão

O script `process_q2.py` analisa a relação entre incidência passada e mortalidade acumulada por milhão de habitantes.

São utilizadas variáveis desfasadas a 14 e 21 dias, assumindo uma latência temporal entre o aumento de casos e o eventual impacto na mortalidade.

São testados dois modelos:

1. Regressão Linear Múltipla.
2. Regressão Polinomial de Grau 2.

A variável CFR não é usada como preditor, por ser matematicamente derivada de casos e óbitos, o que poderia introduzir circularidade na modelação.

O output gerado é:

```text
datasets/outputs_q2.csv
```

## Questão 3 — Taxa de Letalidade

O script `process_q3.py` calcula e compara a Taxa de Letalidade (CFR) de Portugal e da referência mundial.

A CFR é calculada como:

```text
CFR (%) = (total_deaths / total_cases) * 100
```

Para reduzir ruído estatístico nas fases iniciais da pandemia, aplica-se um filtro mínimo de estabilização:

```text
total_cases >= 1000
```

O output gerado é:

```text
datasets/outputs_q3.csv
```

Este ficheiro contém as séries sincronizadas de CFR para Portugal e Mundo.

## Questão 4 — Consolidação de métricas

O script `process_q4.py` consolida as métricas de validação dos modelos aplicados nas Questões 1 e 2.

São calculadas as seguintes métricas:

1. MAE — Mean Absolute Error.
2. RMSE — Root Mean Squared Error.
3. R² — Coeficiente de Determinação.

O output gerado é:

```text
datasets/outputs_q4.csv
```

Este ficheiro permite comparar formalmente o desempenho dos modelos e sustentar a análise crítica do relatório.

## Geração de gráficos

O script `gerar_graficos_relatorio.py` gera automaticamente os gráficos utilizados no relatório final.

São produzidos quatro ficheiros:

```text
graficos/grafico_q1_incidencia.png
graficos/grafico_q2_mortalidade.png
graficos/grafico_q3_cfr.png
graficos/grafico_q4_metricas.png
```

Os gráficos representam:

1. Incidência real vs previsões dos modelos da Q1.
2. Mortalidade real vs previsões dos modelos da Q2.
3. CFR de Portugal vs Mundo.
4. Comparação do RMSE entre modelos.

## Interpretação metodológica

A solução não procura apenas maximizar métricas preditivas. O objetivo é também avaliar criticamente quando um modelo acrescenta valor analítico e quando uma abordagem simples pode ser mais adequada.

Na Q1, a comparação com o Baseline Naïve é essencial, porque séries epidemiológicas suavizadas apresentam forte autocorrelação de curto prazo.

Na Q2, os resultados devem ser interpretados com cautela, uma vez que a mortalidade acumulada por milhão é uma variável cumulativa e não-estacionária. Assim, modelos de regressão estática podem revelar dificuldades de extrapolação temporal.

Na Q3, o filtro de estabilização da CFR reduz distorções associadas aos períodos iniciais de baixo volume de casos.

Na Q4, a consolidação das métricas permite comparar os modelos de forma transversal e fundamentar a discussão crítica final.

## Limitações

A análise apresenta algumas limitações metodológicas:

1. A qualidade dos resultados depende da consistência dos dados originais.
2. Podem existir efeitos de reporte, atrasos administrativos e alterações metodológicas entre países.
3. A previsão da mortalidade acumulada é dificultada pela natureza cumulativa da variável.
4. A Random Forest tem limitações na extrapolação de valores fora do padrão observado no treino.
5. O ARIMA foi aplicado em previsão multietapa, sem atualização diária com os valores reais do conjunto de teste.
6. Os outputs foram operacionalizados em Python, ficando preparados para análise posterior em ferramentas de Business Intelligence ou outros ambientes de visualização.

## Síntese final

Este repositório documenta uma infraestrutura analítica completa, desde a utilização de datasets limpos até à aplicação de modelos preditivos, validação estatística, geração de outputs e produção de gráficos.

A abordagem adotada privilegia transparência metodológica, rastreabilidade dos resultados e análise crítica, aspetos essenciais em sistemas de suporte à decisão aplicados a dados epidemiológicos.
