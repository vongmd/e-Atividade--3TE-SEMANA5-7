"""
process_q4.py

Questão 4 — Consolidação das métricas de validação dos modelos preditivos.

OBJETIVO CIENTÍFICO
-------------------
Consolidar, numa única matriz, as métricas de desempenho dos modelos
implementados nas Questões 1 e 2.

Este script lê:
    - datasets/outputs_q1.csv
    - datasets/outputs_q2.csv

E gera:
    - datasets/outputs_q4.csv

MÉTRICAS CALCULADAS
-------------------
1. MAE  — Mean Absolute Error / Erro Médio Absoluto
2. RMSE — Root Mean Squared Error / Raiz do Erro Quadrático Médio
3. R2   — Coeficiente de Determinação

ALTERAÇÕES FACE À VERSÃO ANTERIOR
---------------------------------
1. Caminhos mais robustos com Path(__file__).
2. Validação explícita da existência dos outputs de Q1 e Q2.
3. Validação das colunas necessárias.
4. Código organizado por funções.
5. Compatibilidade com os nomes das colunas produzidas pelos scripts revistos:
   - process_q1.py
   - process_q2.py
6. Exportação em UTF-8 com BOM, facilitando abertura em Excel.
7. Remoção de qualquer dependência de Qlik ou ferramenta externa.

NOTA METODOLÓGICA
-----------------
A Q4 funciona como etapa de auditoria transversal. O objetivo não é treinar
novos modelos, mas comparar formalmente o desempenho dos modelos já
executados e apoiar a discussão crítica do relatório final.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# 1. Definição robusta de caminhos
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"

Q1_FILE = DATASETS_DIR / "outputs_q1.csv"
Q2_FILE = DATASETS_DIR / "outputs_q2.csv"
OUTPUT_FILE = DATASETS_DIR / "outputs_q4.csv"


# ---------------------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------------------

def validar_ficheiro(caminho):
    """
    Verifica se o ficheiro existe localmente.
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {caminho}\n"
            "Execute primeiro os scripts anteriores necessários."
        )


def normalizar_colunas(df):
    """
    Normaliza nomes de colunas para minúsculas e underscores.

    Esta normalização aumenta a robustez caso os ficheiros CSV tenham
    pequenas diferenças formais nos cabeçalhos.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def validar_colunas(df, colunas_obrigatorias, nome_ficheiro):
    """
    Verifica se todas as colunas necessárias existem no DataFrame.
    """
    colunas_em_falta = [
        coluna for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if colunas_em_falta:
        raise ValueError(
            f"O ficheiro {nome_ficheiro} não contém todas as colunas necessárias. "
            f"Colunas em falta: {colunas_em_falta}"
        )


def calcular_metricas(y_true, y_pred):
    """
    Calcula as métricas formais de avaliação preditiva.

    Remove valores nulos antes do cálculo para evitar erros em situações
    em que alguma previsão contenha NaN.
    """
    dados = pd.DataFrame(
        {
            "real": y_true,
            "previsto": y_pred,
        }
    ).dropna()

    if dados.empty:
        raise ValueError(
            "Não existem pares válidos real/previsto para calcular métricas."
        )

    return {
        "MAE": mean_absolute_error(dados["real"], dados["previsto"]),
        "RMSE": np.sqrt(mean_squared_error(dados["real"], dados["previsto"])),
        "R2": r2_score(dados["real"], dados["previsto"]),
    }


def adicionar_resultado(lista_resultados, questao, modelo, y_true, y_pred):
    """
    Calcula métricas e adiciona uma linha à lista final de resultados.
    """
    metricas = calcular_metricas(y_true, y_pred)

    lista_resultados.append(
        {
            "Questao": questao,
            "Modelo": modelo,
            "MAE": round(metricas["MAE"], 4),
            "RMSE": round(metricas["RMSE"], 4),
            "R2": round(metricas["R2"], 4),
        }
    )


# ---------------------------------------------------------------------
# 3. Validação e leitura dos outputs anteriores
# ---------------------------------------------------------------------

validar_ficheiro(Q1_FILE)
validar_ficheiro(Q2_FILE)

df_q1 = pd.read_csv(Q1_FILE)
df_q2 = pd.read_csv(Q2_FILE)

df_q1 = normalizar_colunas(df_q1)
df_q2 = normalizar_colunas(df_q2)


# ---------------------------------------------------------------------
# 4. Validação das colunas da Q1
# ---------------------------------------------------------------------

colunas_q1 = [
    "new_cases_7_day_avg_right",
    "pred_baseline",
    "pred_arima",
    "pred_rf",
]

validar_colunas(
    df_q1,
    colunas_q1,
    "outputs_q1.csv",
)


# ---------------------------------------------------------------------
# 5. Validação das colunas da Q2
# ---------------------------------------------------------------------

colunas_q2 = [
    "total_deaths_per_million",
    "pred_regressao_linear",
    "pred_regressao_polinomial_g2",
]

validar_colunas(
    df_q2,
    colunas_q2,
    "outputs_q2.csv",
)


# ---------------------------------------------------------------------
# 6. Cálculo das métricas consolidadas
# ---------------------------------------------------------------------

resultados = []

# Questão 1 — Incidência diária suavizada
adicionar_resultado(
    resultados,
    questao="Q1 - Incidência",
    modelo="Baseline Naïve (Lag 1)",
    y_true=df_q1["new_cases_7_day_avg_right"],
    y_pred=df_q1["pred_baseline"],
)

adicionar_resultado(
    resultados,
    questao="Q1 - Incidência",
    modelo="ARIMA (1,1,1)",
    y_true=df_q1["new_cases_7_day_avg_right"],
    y_pred=df_q1["pred_arima"],
)

adicionar_resultado(
    resultados,
    questao="Q1 - Incidência",
    modelo="Random Forest Regressor",
    y_true=df_q1["new_cases_7_day_avg_right"],
    y_pred=df_q1["pred_rf"],
)

# Questão 2 — Mortalidade acumulada por milhão
adicionar_resultado(
    resultados,
    questao="Q2 - Mortalidade",
    modelo="Regressão Linear Múltipla",
    y_true=df_q2["total_deaths_per_million"],
    y_pred=df_q2["pred_regressao_linear"],
)

adicionar_resultado(
    resultados,
    questao="Q2 - Mortalidade",
    modelo="Regressão Polinomial Grau 2",
    y_true=df_q2["total_deaths_per_million"],
    y_pred=df_q2["pred_regressao_polinomial_g2"],
)


# ---------------------------------------------------------------------
# 7. Criação e exportação da matriz final
# ---------------------------------------------------------------------

df_resultados = pd.DataFrame(resultados)

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

df_resultados.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\nMétricas consolidadas — Q4")
print(df_resultados.to_string(index=False))

print(f"\nFicheiro Q4 exportado com sucesso para: {OUTPUT_FILE}")