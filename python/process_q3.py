"""
process_q3.py

Questão 3 — Dinâmica e estabilização da Taxa de Letalidade (CFR).

OBJETIVO CIENTÍFICO
-------------------
Analisar a evolução da Taxa de Letalidade (Case Fatality Rate — CFR) em
Portugal e na referência mundial, após remoção dos períodos iniciais com
baixo volume acumulado de casos.

A CFR é calculada como:

    CFR (%) = (total_deaths / total_cases) * 100

JUSTIFICAÇÃO METODOLÓGICA
-------------------------
Nas fases iniciais de uma epidemia, quando o número acumulado de casos é
muito reduzido, pequenas variações no número de óbitos podem gerar picos
percentuais artificiais. Por isso, aplica-se um filtro mínimo de estabilização:

    total_cases >= 1000

Este procedimento reduz ruído matemático inicial e permite interpretar a
CFR numa fase mais estável da série temporal.

ALTERAÇÕES FACE À VERSÃO ANTERIOR
---------------------------------
1. Caminhos mais robustos com Path(__file__).
2. Validação explícita dos ficheiros de entrada.
3. Validação das colunas obrigatórias.
4. Cálculo controlado da CFR a partir de total_deaths e total_cases.
5. Remoção de referências a integração Qlik.
6. Exportação em UTF-8 com BOM para maior compatibilidade com Excel.
7. Manutenção dos nomes das colunas:
   - cfr_portugal
   - cfr_mundo

Estes nomes são preservados para garantir compatibilidade com o script
gerar_graficos_relatorio.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Definição robusta de caminhos
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"

PAISES_FILE = DATASETS_DIR / "covid_paises_limpo.csv"
GLOBAL_FILE = DATASETS_DIR / "covid_global_limpo.csv"
OUTPUT_FILE = DATASETS_DIR / "outputs_q3.csv"


# ---------------------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------------------

def normalizar_colunas(df):
    """
    Normaliza os nomes das colunas para minúsculas e underscores.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def validar_ficheiro(caminho):
    """
    Confirma se o ficheiro existe localmente.
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {caminho}\n"
            "Confirme que os datasets limpos existem localmente na pasta datasets/."
        )


def validar_colunas(df, colunas_obrigatorias, nome_dataset):
    """
    Verifica se todas as colunas obrigatórias existem no DataFrame.
    """
    colunas_em_falta = [
        coluna for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if colunas_em_falta:
        raise ValueError(
            f"O dataset {nome_dataset} não contém todas as colunas necessárias. "
            f"Colunas em falta: {colunas_em_falta}"
        )


def preparar_cfr(df, nome_serie):
    """
    Calcula a CFR a partir de total_deaths e total_cases.

    A CFR é expressa em percentagem.
    Valores inválidos, como divisões por zero, são convertidos em NaN.
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_cases"] = pd.to_numeric(df["total_cases"], errors="coerce")
    df["total_deaths"] = pd.to_numeric(df["total_deaths"], errors="coerce")

    # Evita divisão por zero.
    df["cfr"] = np.where(
        df["total_cases"] > 0,
        (df["total_deaths"] / df["total_cases"]) * 100,
        np.nan,
    )

    # Aplica filtro de estabilização volumétrica.
    df = df[df["total_cases"] >= 1000].copy()

    # Remove registos sem data ou sem CFR válida.
    df = df.dropna(subset=["date", "cfr"])

    if df.empty:
        raise ValueError(
            f"A série {nome_serie} ficou vazia após o filtro total_cases >= 1000."
        )

    return df


def filtrar_geografia(df, valor_geografia, nome_dataset):
    """
    Filtra uma geografia específica, usando a coluna country.

    Esta função é usada para Portugal.
    """
    validar_colunas(df, ["country"], nome_dataset)

    df_filtrado = (
        df[df["country"].astype(str).str.lower() == valor_geografia.lower()]
        .copy()
    )

    if df_filtrado.empty:
        raise ValueError(
            f"Não foram encontrados registos para {valor_geografia} "
            f"no dataset {nome_dataset}."
        )

    return df_filtrado


def preparar_referencia_mundial(df_global):
    """
    Prepara a série de referência mundial.

    O ficheiro covid_global_limpo.csv pode conter apenas a série mundial
    ou pode incluir uma coluna country com o valor World. Esta função lida
    com as duas situações.
    """
    df = df_global.copy()

    if "country" in df.columns:
        df_world = df[df["country"].astype(str).str.lower() == "world"].copy()

        if not df_world.empty:
            return df_world

    # Caso o ficheiro global já contenha apenas a série mundial, não filtra.
    return df


# ---------------------------------------------------------------------
# 3. Carregamento dos datasets
# ---------------------------------------------------------------------

validar_ficheiro(PAISES_FILE)
validar_ficheiro(GLOBAL_FILE)

df_paises = pd.read_csv(PAISES_FILE)
df_global = pd.read_csv(GLOBAL_FILE)

df_paises = normalizar_colunas(df_paises)
df_global = normalizar_colunas(df_global)

colunas_base = [
    "date",
    "total_cases",
    "total_deaths",
]

validar_colunas(df_paises, colunas_base + ["country"], "covid_paises_limpo.csv")
validar_colunas(df_global, colunas_base, "covid_global_limpo.csv")


# ---------------------------------------------------------------------
# 4. Preparação da série de Portugal
# ---------------------------------------------------------------------

df_pt = filtrar_geografia(
    df_paises,
    valor_geografia="Portugal",
    nome_dataset="covid_paises_limpo.csv",
)

df_pt_estavel = preparar_cfr(
    df_pt,
    nome_serie="Portugal",
)

df_pt_cfr = df_pt_estavel[
    [
        "date",
        "total_cases",
        "total_deaths",
        "cfr",
    ]
].rename(
    columns={
        "total_cases": "total_cases_portugal",
        "total_deaths": "total_deaths_portugal",
        "cfr": "cfr_portugal",
    }
)


# ---------------------------------------------------------------------
# 5. Preparação da referência mundial
# ---------------------------------------------------------------------

df_world = preparar_referencia_mundial(df_global)

df_world_estavel = preparar_cfr(
    df_world,
    nome_serie="Mundo",
)

df_world_cfr = df_world_estavel[
    [
        "date",
        "total_cases",
        "total_deaths",
        "cfr",
    ]
].rename(
    columns={
        "total_cases": "total_cases_mundo",
        "total_deaths": "total_deaths_mundo",
        "cfr": "cfr_mundo",
    }
)


# ---------------------------------------------------------------------
# 6. Sincronização temporal das duas séries
# ---------------------------------------------------------------------
# Faz-se uma junção por data para garantir que Portugal e Mundo são
# comparados apenas nos dias em que ambas as séries têm valores válidos.

df_q3 = pd.merge(
    df_pt_cfr,
    df_world_cfr,
    on="date",
    how="inner",
)

df_q3 = df_q3.sort_values("date").reset_index(drop=True)

if df_q3.empty:
    raise ValueError(
        "A matriz Q3 ficou vazia após a sincronização temporal das séries."
    )


# ---------------------------------------------------------------------
# 7. Exportação do output analítico
# ---------------------------------------------------------------------
# Este ficheiro suporta:
# - análise comparativa da CFR Portugal vs Mundo;
# - geração do gráfico da Questão 3;
# - sustentação da discussão crítica no relatório final.

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

df_q3.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(
    "Dados Q3 processados com sucesso.\n"
    f"Registos sincronizados após estabilização: {len(df_q3)}\n"
    f"Ficheiro Q3 exportado para: {OUTPUT_FILE}"
)