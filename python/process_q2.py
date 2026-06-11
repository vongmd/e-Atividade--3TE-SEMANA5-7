"""
process_q2.py

Questão 2 — Modelação da mortalidade acumulada por milhão de habitantes.

OBJETIVO CIENTÍFICO
-------------------
Avaliar se a incidência registada em momentos anteriores permite explicar
a evolução da mortalidade acumulada por milhão de habitantes em Portugal.

A lógica metodológica assenta na existência de uma latência temporal entre:
    - infeção / aumento de casos;
    - agravamento clínico;
    - eventual registo de óbito.

Por isso, são criadas variáveis desfasadas a 14 e 21 dias.

MODELOS TESTADOS
----------------
1. Regressão Linear Múltipla
2. Regressão Polinomial de Grau 2

ALTERAÇÕES FACE À VERSÃO ANTERIOR
---------------------------------
1. Caminhos mais robustos com Path(__file__).
2. Validação explícita da existência do ficheiro de entrada.
3. Validação das colunas obrigatórias.
4. Escolha controlada da variável de incidência:
   - dá prioridade a new_cases_per_million;
   - se não existir, usa new_cases_7_day_avg_right;
   - se também não existir, usa new_cases.
5. Exclusão explícita da CFR dos preditores, evitando circularidade matemática.
6. Código organizado por secções, com comentários metodológicos.
7. Exportação em UTF-8 com BOM, mais compatível com Excel em Windows.

NOTA METODOLÓGICA
-----------------
A variável de destino é cumulativa e, por natureza, não-estacionária. Assim,
é expectável que modelos de regressão estática possam ter mau desempenho
na extrapolação temporal. Este resultado não invalida o exercício; pelo
contrário, é relevante para a análise crítica do suporte à decisão.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# ---------------------------------------------------------------------
# 1. Definição robusta de caminhos
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"

INPUT_FILE = DATASETS_DIR / "covid_paises_limpo.csv"
OUTPUT_FILE = DATASETS_DIR / "outputs_q2.csv"


# ---------------------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------------------

def normalizar_colunas(df):
    """
    Normaliza os nomes das colunas para minúsculas e underscores.

    Esta função reduz o risco de erro caso existam pequenas diferenças
    formais nos cabeçalhos do ficheiro CSV.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def validar_colunas(df, colunas_obrigatorias):
    """
    Valida se todas as colunas necessárias existem no DataFrame.
    """
    colunas_em_falta = [
        coluna for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if colunas_em_falta:
        raise ValueError(
            "O dataset não contém todas as colunas necessárias. "
            f"Colunas em falta: {colunas_em_falta}"
        )


def escolher_variavel_incidencia(df):
    """
    Seleciona a variável de incidência mais adequada disponível no dataset.

    Ordem de preferência:
    1. new_cases_per_million
       - Preferível por estar normalizada pela população.
    2. new_cases_7_day_avg_right
       - Útil por estar suavizada por média móvel.
    3. new_cases
       - Usada apenas se as alternativas anteriores não existirem.

    Esta flexibilidade permite que o script funcione mesmo quando o dataset
    limpo contém pequenas diferenças na seleção de atributos.
    """
    candidatas = [
        "new_cases_per_million",
        "new_cases_7_day_avg_right",
        "new_cases",
    ]

    for coluna in candidatas:
        if coluna in df.columns:
            return coluna

    raise ValueError(
        "Não foi encontrada nenhuma variável de incidência adequada. "
        "Esperava-se uma das seguintes colunas: "
        "new_cases_per_million, new_cases_7_day_avg_right ou new_cases."
    )


def calcular_metricas(y_true, y_pred):
    """
    Calcula as métricas formais de avaliação dos modelos.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------
# 3. Carregamento e preparação dos dados
# ---------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Ficheiro não encontrado: {INPUT_FILE}\n"
        "Confirme que o ficheiro covid_paises_limpo.csv existe localmente "
        "na pasta datasets/."
    )

df = pd.read_csv(INPUT_FILE)
df = normalizar_colunas(df)

# A variável de incidência pode variar conforme o pipeline anterior.
variavel_incidencia = escolher_variavel_incidencia(df)

validar_colunas(
    df,
    [
        "country",
        "date",
        "total_deaths_per_million",
        variavel_incidencia,
    ],
)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_deaths_per_million"] = pd.to_numeric(
    df["total_deaths_per_million"],
    errors="coerce",
)
df[variavel_incidencia] = pd.to_numeric(
    df[variavel_incidencia],
    errors="coerce",
)

# Filtragem dos dados de Portugal.
df_pt = (
    df[df["country"].astype(str).str.lower() == "portugal"]
    .sort_values("date")
    .copy()
)

if df_pt.empty:
    raise ValueError("Não foram encontrados registos para Portugal no dataset.")

# Seleção das colunas necessárias à Q2.
df_q2 = df_pt[
    [
        "date",
        variavel_incidencia,
        "total_deaths_per_million",
    ]
].dropna().copy()

if len(df_q2) < 100:
    raise ValueError(
        "A série temporal contém poucos registos válidos para treino e teste."
    )


# ---------------------------------------------------------------------
# 4. Engenharia de atributos por desfasamento temporal
# ---------------------------------------------------------------------
# A hipótese analítica é que a mortalidade observada num determinado dia
# pode estar relacionada com a incidência registada 14 e 21 dias antes.

df_q2["lag_14_new_cases"] = df_q2[variavel_incidencia].shift(14)
df_q2["lag_21_new_cases"] = df_q2[variavel_incidencia].shift(21)

# Remoção das primeiras linhas, que ficam sem valores de lag.
df_q2 = df_q2.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------
# 5. Definição de variáveis explicativas e variável-alvo
# ---------------------------------------------------------------------
# IMPORTANTE:
# A CFR não é usada como preditor, porque é matematicamente derivada de
# casos e óbitos. Usá-la aqui criaria circularidade e enviesamento.

features = [
    "lag_14_new_cases",
    "lag_21_new_cases",
]

target = "total_deaths_per_million"


# ---------------------------------------------------------------------
# 6. Divisão cronológica treino/teste
# ---------------------------------------------------------------------
# Tal como na Q1, a divisão é temporal e não aleatória.
# Isto impede a fuga de informação entre passado e futuro.

split_idx = int(len(df_q2) * 0.8)

treino = df_q2.iloc[:split_idx].copy()
teste = df_q2.iloc[split_idx:].copy()

if treino.empty or teste.empty:
    raise ValueError("A divisão treino/teste gerou subconjuntos vazios.")

X_train = treino[features]
y_train = treino[target]

X_test = teste[features]
y_test = teste[target]

print(
    "Dados Q2 preparados com sucesso.\n"
    f"Variável de incidência utilizada: {variavel_incidencia}\n"
    f"Registos de treino: {len(treino)}\n"
    f"Registos de teste: {len(teste)}"
)


# ---------------------------------------------------------------------
# 7. Modelo 1 — Regressão Linear Múltipla
# ---------------------------------------------------------------------

modelo_linear = LinearRegression()
modelo_linear.fit(X_train, y_train)

teste["pred_regressao_linear"] = modelo_linear.predict(X_test)


# ---------------------------------------------------------------------
# 8. Modelo 2 — Regressão Polinomial de Grau 2
# ---------------------------------------------------------------------
# A regressão polinomial permite testar relações não-lineares simples.
# Contudo, em variáveis cumulativas e não-estacionárias, pode agravar
# problemas de extrapolação.

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

modelo_polinomial = LinearRegression()
modelo_polinomial.fit(X_train_poly, y_train)

teste["pred_regressao_polinomial_g2"] = modelo_polinomial.predict(X_test_poly)


# ---------------------------------------------------------------------
# 9. Avaliação dos modelos
# ---------------------------------------------------------------------

modelos = {
    "Regressão Linear Múltipla": "pred_regressao_linear",
    "Regressão Polinomial Grau 2": "pred_regressao_polinomial_g2",
}

metricas = {}

print("\n--- DESEMPENHO DOS MODELOS NO CONJUNTO DE TESTE — Q2 ---")

for nome_modelo, coluna_predicao in modelos.items():
    metricas[nome_modelo] = calcular_metricas(
        y_test,
        teste[coluna_predicao],
    )

    print(f"\n{nome_modelo}")
    print(f"MAE : {metricas[nome_modelo]['MAE']:.2f}")
    print(f"RMSE: {metricas[nome_modelo]['RMSE']:.2f}")
    print(f"R²  : {metricas[nome_modelo]['R2']:.4f}")


# ---------------------------------------------------------------------
# 10. Exportação do output analítico
# ---------------------------------------------------------------------
# Este ficheiro suporta:
# - comparação entre mortalidade real e prevista;
# - geração de gráficos;
# - cálculo consolidado de métricas na Q4;
# - análise crítica no relatório final.

output_df = teste[
    [
        "date",
        target,
        "pred_regressao_linear",
        "pred_regressao_polinomial_g2",
    ]
].copy()

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

output_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nFicheiro Q2 exportado com sucesso para: {OUTPUT_FILE}")