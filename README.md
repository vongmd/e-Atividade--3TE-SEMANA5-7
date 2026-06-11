# Infraestrutura de Modelação Preditiva e Suporte à Decisão — COVID-19

## Unidade Curricular: Sistemas de Business Intelligence — Semanas 13-15

Este repositório contém a solução modular de engenharia de dados, modelação estatística e validação preditiva desenvolvida em Python no âmbito da fase final da e-Atividade. O objetivo do projeto é processar dados históricos da pandemia COVID-19, treinar e comparar modelos preditivos, extrair métricas de validação e gerar matrizes analíticas e gráficos destinados à análise crítica e ao suporte à decisão.

A solução assenta numa arquitetura reprodutível em Python, privilegiando a auditabilidade dos modelos, a separação entre dados de entrada, scripts de processamento, outputs analíticos e elementos gráficos de apoio ao relatório final.

---

## 1. Objetivo da Solução

A infraestrutura desenvolvida tem como finalidade responder às questões de investigação definidas na fase anterior do trabalho, evoluindo da análise exploratória e pré-tratamento de dados para a aplicação prática de modelos preditivos.

A solução permite:

* reutilizar os datasets limpos resultantes do pipeline de pré-tratamento anterior;
* aplicar modelos estatísticos e algoritmos de aprendizagem automática;
* comparar previsões com valores reais observados;
* calcular métricas formais de desempenho;
* gerar ficheiros CSV estruturados para análise posterior;
* produzir gráficos destinados ao relatório final;
* apoiar uma discussão crítica sobre a utilidade e limitações dos modelos preditivos em contexto epidemiológico.

---

## 2. Estrutura do Repositório

A estrutura recomendada do projeto é a seguinte:

```text
e-Atividade--3TE-SEMANA5-7-main/
│
├── datasets/
│   ├── covid_global_limpo.csv
│   ├── outputs_q1.csv
│   ├── outputs_q2.csv
│   ├── outputs_q3.csv
│   └── outputs_q4.csv
│
├── python/
│   ├── process_q1.py
│   ├── process_q2.py
│   ├── process_q3.py
│   ├── process_q4.py
│   └── gerar_graficos_relatorio.py
│
├── graficos/
│   ├── grafico_q1_incidencia.png
│   ├── grafico_q2_mortalidade.png
│   └── grafico_q3_cfr.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

Nota: o ficheiro `datasets/covid_paises_limpo.csv` não é incluído no repositório por exceder o limite recomendado para versionamento direto no GitHub. Deve existir localmente para que os scripts possam ser executados.

---

## 3. Artefactos Criados

### 3.1. Scripts de Modelação (`python/`)

O projeto foi estruturado de forma modular, garantindo o isolamento lógico entre as diferentes questões de investigação.

* `process_q1.py`: executa a modelação da incidência diária suavizada em Portugal, comparando uma linha de base Naïve, um modelo ARIMA e um algoritmo Random Forest Regressor.
* `process_q2.py`: implementa a engenharia de atributos por desfasamento temporal e ajusta modelos de regressão para a mortalidade acumulada por milhão de habitantes.
* `process_q3.py`: aplica filtros de estabilização volumétrica sobre a Taxa de Letalidade (CFR), reduzindo o ruído estatístico inicial.
* `process_q4.py`: consolida as métricas de erro dos modelos, funcionando como guião de auditoria quantitativa.
* `gerar_graficos_relatorio.py`: lê as matrizes analíticas geradas e exporta os gráficos utilizados no relatório final.

### 3.2. Datasets Analíticos Criados (`datasets/`)

Os ficheiros seguintes são o produto final do processamento em Python e constituem as matrizes analíticas usadas para validação dos modelos, geração dos gráficos e sustentação da análise crítica do relatório:

* `outputs_q1.csv`: matriz temporal que emparelha a curva real de novos casos com as previsões geradas pelos modelos aplicados na Questão 1.
* `outputs_q2.csv`: série temporal com os valores reais de mortalidade por milhão e as respetivas projeções de regressão da Questão 2.
* `outputs_q3.csv`: matriz cronológica sincronizada entre a Taxa de Letalidade de Portugal e a referência global após aplicação do filtro de estabilização.
* `outputs_q4.csv`: tabela estruturada de métricas de desempenho, incluindo MAE, RMSE e R².

---

## 4. Datasets Base Não Versionados no GitHub

Por razões de dimensão e boas práticas de gestão de repositórios, o ficheiro `datasets/covid_paises_limpo.csv` não é incluído no GitHub, uma vez que excede o limite recomendado para ficheiros versionados diretamente no repositório.

Este ficheiro é necessário para a execução dos scripts `process_q1.py`, `process_q2.py` e `process_q3.py`. Antes da execução da infraestrutura preditiva, o utilizador deve garantir localmente a existência dos seguintes ficheiros na pasta `datasets/`:

```text
datasets/covid_paises_limpo.csv
datasets/covid_global_limpo.csv
```

O ficheiro `covid_paises_limpo.csv` resulta do pipeline de extração, limpeza e pré-tratamento desenvolvido na fase anterior do trabalho, correspondente às semanas 11-12. A sua exclusão do repositório não compromete a reprodutibilidade da solução, desde que o dataset seja recuperado a partir da fase anterior ou regenerado através do pipeline de pré-tratamento.

Os ficheiros `outputs_q1.csv`, `outputs_q2.csv`, `outputs_q3.csv` e `outputs_q4.csv` são mantidos no repositório, pois representam matrizes analíticas finais de menor dimensão, diretamente utilizadas para validação, visualização e análise crítica.

---

## 5. Lógica de Processamento por Questão de Investigação

### 5.1. Questão 1 — Previsão da Incidência Diária Suavizada

O script `process_q1.py` isola a variável `new_cases_7_day_avg_right`, correspondente à média móvel de 7 dias dos novos casos, com o objetivo de reduzir a variabilidade artificial provocada por atrasos de reporte, especialmente o efeito de fim de semana.

A série temporal é dividida de forma cronológica:

* 80% dos dados para treino;
* 20% dos dados para teste.

Esta divisão evita a fuga de informação (*data leakage*), uma vez que impede que valores futuros sejam usados no treino dos modelos.

São comparadas três abordagens:

1. **Baseline Naïve** — assume que o valor previsto para o dia seguinte corresponde ao valor observado no dia anterior.
2. **ARIMA (1,1,1)** — modelo estatístico autorregressivo integrado de média móvel.
3. **Random Forest Regressor** — algoritmo de aprendizagem supervisionada baseado em árvores de decisão, alimentado por variáveis desfasadas de 1, 7 e 14 dias.

O output gerado é:

```text
datasets/outputs_q1.csv
```

---

### 5.2. Questão 2 — Modelação da Mortalidade Acumulada por Milhão

O script `process_q2.py` modela a variável `total_deaths_per_million`, procurando avaliar se a incidência registada em dias anteriores permite explicar a evolução da mortalidade acumulada por milhão de habitantes.

Para representar a latência biológica entre infeção e eventual desfecho de óbito, são criadas variáveis independentes com desfasamento temporal:

* `lag_14_new_cases`;
* `lag_21_new_cases`.

A Taxa de Letalidade (CFR) não é utilizada como preditor, de forma a evitar circularidade matemática, uma vez que esse indicador deriva da relação entre casos e óbitos acumulados.

São testados dois modelos:

1. **Regressão Linear Múltipla**;
2. **Regressão Polinomial de Grau 2**.

O output gerado é:

```text
datasets/outputs_q2.csv
```

---

### 5.3. Questão 3 — Estabilização da Taxa de Letalidade (CFR)

O script `process_q3.py` analisa a evolução da Taxa de Letalidade (*Case Fatality Rate*) em Portugal e no mundo.

A CFR é calculada a partir da relação entre óbitos acumulados e casos acumulados. Contudo, nas fases iniciais da pandemia, valores muito baixos de casos acumulados geram distorções matemáticas significativas. Para reduzir esse ruído, o script aplica um filtro que remove todos os registos em que o número acumulado de casos é inferior a 1.000.

Critério aplicado:

```text
total_cases >= 1000
```

Este procedimento permite analisar a CFR apenas a partir de uma fase de maior estabilidade estatística.

O output gerado é:

```text
datasets/outputs_q3.csv
```

---

### 5.4. Questão 4 — Auditoria e Validação dos Modelos

O script `process_q4.py` consolida os resultados das questões anteriores e calcula as principais métricas de avaliação dos modelos.

As métricas utilizadas são:

* **MAE** — Erro Médio Absoluto;
* **RMSE** — Raiz do Erro Quadrático Médio;
* **R²** — Coeficiente de Determinação.

O objetivo desta etapa é comparar o desempenho dos modelos e identificar se a sua complexidade acrescenta, ou não, valor preditivo face a linhas de base simples.

O output gerado é:

```text
datasets/outputs_q4.csv
```

---

## 6. Instruções de Instalação

Recomenda-se a utilização de um ambiente virtual Python.

### 6.1. Criação do Ambiente Virtual

Na raiz do projeto:

```bash
python -m venv .venv
```

Ativação em Windows:

```bash
.venv\Scripts\activate
```

Ativação em macOS/Linux:

```bash
source .venv/bin/activate
```

### 6.2. Instalação das Dependências

```bash
pip install -r requirements.txt
```

Caso o ficheiro `requirements.txt` ainda não exista, pode ser criado com as dependências principais:

```text
pandas
numpy
matplotlib
scikit-learn
statsmodels
```

---

## 7. Instruções de Execução

Antes de executar os scripts de modelação, deve confirmar-se que os datasets base existem localmente na pasta `datasets/`:

```text
datasets/covid_paises_limpo.csv
datasets/covid_global_limpo.csv
```

Caso o ficheiro `covid_paises_limpo.csv` não exista localmente, deve ser recuperado a partir da fase anterior do trabalho ou regenerado através do pipeline de extração e pré-tratamento.

Para reprocessar a infraestrutura preditiva e atualizar os ficheiros de dados e relatórios visuais, os scripts devem ser executados sequencialmente a partir da pasta `python/`:

```bash
cd python

python process_q1.py
python process_q2.py
python process_q3.py
python process_q4.py
python gerar_graficos_relatorio.py
```

Após a execução, os ficheiros de saída são gerados ou atualizados na pasta `datasets/`:

```text
outputs_q1.csv
outputs_q2.csv
outputs_q3.csv
outputs_q4.csv
```

Os gráficos são exportados para a pasta:

```text
graficos/
```

---

## 8. Outputs Esperados

Após a execução completa, devem estar disponíveis os seguintes artefactos:

```text
datasets/outputs_q1.csv
datasets/outputs_q2.csv
datasets/outputs_q3.csv
datasets/outputs_q4.csv

graficos/grafico_q1_incidencia.png
graficos/grafico_q2_mortalidade.png
graficos/grafico_q3_cfr.png
```

Estes ficheiros constituem a camada de evidência quantitativa entre a modelação estatística desenvolvida em Python e a análise crítica apresentada no relatório final.

---

## 9. Interpretação dos Resultados

A infraestrutura desenvolvida permite demonstrar que a avaliação crítica dos modelos é tão relevante quanto a sua implementação técnica.

Em particular:

* a linha de base Naïve pode superar modelos mais complexos quando a série temporal apresenta forte autocorrelação de curto prazo;
* modelos ARIMA podem perder desempenho quando usados em previsão direta multietapa sem atualização contínua com novos valores reais;
* algoritmos baseados em árvores, como Random Forest, apresentam limitações quando necessitam de extrapolar valores fora do intervalo observado no treino;
* modelos de regressão estática podem revelar fraco desempenho em variáveis cumulativas e não estacionárias, como a mortalidade acumulada por milhão;
* a estabilização da CFR através de um limiar mínimo de casos acumulados reduz ruído matemático e melhora a qualidade interpretativa do indicador.

---

## 10. Boas Práticas de Versionamento

O repositório deve versionar:

* scripts Python;
* outputs analíticos finais de menor dimensão;
* gráficos finais;
* README;
* ficheiros de configuração;
* relatório final, caso aplicável.

O repositório não deve versionar:

* datasets brutos de grande dimensão;
* ficheiros CSV superiores ao limite recomendado pelo GitHub;
* ambientes virtuais;
* caches Python;
* ficheiros temporários.

Exemplo de `.gitignore` recomendado:

```gitignore
# Datasets grandes locais
datasets/covid_paises_limpo.csv
datasets/covid_casos_mortes_bruto.csv
datasets/*.zip

# Ambientes virtuais e cache Python
.venv/
venv/
__pycache__/
*.pyc

# Ficheiros temporários
*.tmp
.DS_Store
Thumbs.db
```

---

## 11. Limitações Conhecidas

A solução apresenta algumas limitações que devem ser consideradas na interpretação dos resultados:

* os dados epidemiológicos estão sujeitos a assimetrias internacionais de reporte;
* a frequência de testagem e os critérios de notificação variam entre países;
* modelos treinados em dados históricos podem não generalizar adequadamente para novas variantes ou alterações estruturais na política sanitária;
* variáveis cumulativas tendem a ser não estacionárias, exigindo cuidados adicionais na modelação;
* a ausência de variáveis demográficas, hospitalares ou vacinais mais detalhadas limita a capacidade explicativa de alguns modelos.

Estas limitações não invalidam a solução, mas reforçam a necessidade de interpretar os modelos como instrumentos de apoio à decisão e não como mecanismos automáticos de decisão.

---

## 12. Síntese Final

Este projeto consolida o percurso desenvolvido ao longo das várias fases da e-Atividade. A fase inicial fundamentou a utilização de soluções Big Data e sistemas de Business Intelligence no contexto da pandemia. A fase seguinte operacionalizou a recolha, limpeza e pré-tratamento dos dados. A presente fase final aplica modelos preditivos, valida os seus resultados e produz evidência quantitativa para discussão crítica.

O principal contributo da solução reside na demonstração de que o valor analítico dos dados não depende apenas da sofisticação dos algoritmos utilizados, mas da capacidade de preparar, validar, comparar e interpretar criticamente os resultados obtidos.
