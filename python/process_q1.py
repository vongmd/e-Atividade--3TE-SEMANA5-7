"""
process_q1.py

Questão 1 — Modelação e previsão da incidência diária suavizada em Portugal.

OBJETIVO CIENTÍFICO
-------------------
Responder à questão de investigação associada à previsão da incidência diária
suavizada da COVID-19 em Portugal, comparando modelos simples e modelos
mais complexos.

A lógica científica mantém-se igual à versão anterior:
    - variável-alvo: new_cases_7_day_avg_right
    - divisão cronológica: 80% treino / 20% teste
    - modelos comparados:
        1. Baseline Naïve
        2. ARIMA (1,1,1)
        3. Random Forest Regressor
    - output final: datasets/outputs_q1.csv

ALTERAÇÕES FACE À VERSÃO ANTERIOR
---------------------------------
1. Caminhos mais robustos:
   - Antes: dependiam de executar o script dentro da pasta python/.
   - Agora: usam Path(__file__), permitindo execução mais segura.

2. Validação do dataset:
   - Verifica se o ficheiro existe.
   - Verifica se as colunas obrigatórias existem.
   - Verifica se existem dados para Portugal.
   - Verifica se há registos suficientes para treino/teste.

3. Código mais legível e auditável:
   - Organizado por secções numeradas.
   - Funções auxiliares para normalização, validação e métricas.

4. Exportação mais compatível:
   - Usa encoding="utf-8-sig", facilitando abertura em Excel no Windows.

5. Random Forest otimizado:
   - Mantém os mesmos parâmetros essenciais.
   - Acrescenta n_jobs=-1 para usar os núcleos disponíveis do processador.

NOTA METODOLÓGICA
-----------------
A avaliação é feita de forma cronológica, evitando data leakage. O Baseline Naïve
e o Random Forest são avaliados numa lógica de previsão de curto prazo com
variáveis de atraso temporal. O ARIMA é aplicado em previsão multietapa sobre
o horizonte de teste, sem atualização diária com novos valores reais.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


# ---------------------------------------------------------------------
# 1. Definição robusta de caminhos
# ---------------------------------------------------------------------
# ALTERAÇÃO:
# Na versão anterior, os caminhos dependiam de o script ser executado
# exatamente dentro da pasta python/.
#
# Agora, BASE_DIR identifica automaticamente a raiz do projeto,
# independentemente da localização a partir da qual o script é executado.

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"

INPUT_FILE = DATASETS_DIR / "covid_paises_limpo.csv"
OUTPUT_FILE = DATASETS_DIR / "outputs_q1.csv"


# ---------------------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------------------

def normalizar_colunas(df):
    """
    Normaliza os nomes das colunas.

    Justificação:
    Alguns datasets podem conter espaços, maiúsculas ou pequenas variações
    nos nomes das colunas. Esta normalização reduz o risco de erro por
    inconsistência formal.
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
    Verifica se todas as colunas necessárias existem no dataset.

    ALTERAÇÃO:
    A versão anterior assumia implicitamente que todas as colunas existiam.
    Esta validação torna o erro mais explícito e mais fácil de corrigir.
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


def calcular_metricas(y_true, y_pred):
    """
    Calcula as métricas formais de avaliação preditiva.

    Métricas:
    - MAE: erro médio absoluto.
    - RMSE: penaliza mais fortemente erros elevados.
    - R2: mede a capacidade explicativa do modelo face à variabilidade real.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------
# 3. Carregamento e preparação dos dados
# ---------------------------------------------------------------------

# Validação da existência do ficheiro de entrada.
# Este ficheiro não deve ser versionado no GitHub se for demasiado grande,
# mas tem de existir localmente na pasta datasets/.
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Ficheiro não encontrado: {INPUT_FILE}\n"
        "Confirme que o ficheiro covid_paises_limpo.csv existe localmente "
        "na pasta datasets/."
    )

# Leitura do dataset.
df = pd.read_csv(INPUT_FILE)

# Normalização dos nomes das colunas.
df = normalizar_colunas(df)

# Validação das colunas necessárias para a Questão 1.
validar_colunas(
    df,
    [
        "country",
        "date",
        "new_cases_7_day_avg_right",
    ],
)

# Conversão da data para formato datetime.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Conversão da variável-alvo para formato numérico.
df["new_cases_7_day_avg_right"] = pd.to_numeric(
    df["new_cases_7_day_avg_right"],
    errors="coerce",
)

# Filtragem da série de Portugal.
df_pt = (
    df[df["country"].astype(str).str.lower() == "portugal"]
    .sort_values("date")
    .copy()
)

# Validação da existência de dados para Portugal.
if df_pt.empty:
    raise ValueError("Não foram encontrados registos para Portugal no dataset.")

# Seleção da variável temporal e da variável-alvo.
df_q1 = df_pt[
    [
        "date",
        "new_cases_7_day_avg_right",
    ]
].dropna().copy()

# Validação mínima de dimensão da série temporal.
# Evita correr modelos sobre séries demasiado pequenas.
if len(df_q1) < 100:
    raise ValueError(
        "A série temporal de Portugal contém poucos registos válidos "
        "para treino e validação."
    )


# ---------------------------------------------------------------------
# 4. Engenharia de atributos temporais
# ---------------------------------------------------------------------
# Mantém-se a lógica da versão anterior:
# criação de variáveis de atraso temporal para alimentar o Random Forest
# e para construir a linha de base Naïve.

df_q1["lag_1"] = df_q1["new_cases_7_day_avg_right"].shift(1)
df_q1["lag_7"] = df_q1["new_cases_7_day_avg_right"].shift(7)
df_q1["lag_14"] = df_q1["new_cases_7_day_avg_right"].shift(14)

# Remoção das primeiras linhas sem valores de lag.
df_q1 = df_q1.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------
# 5. Divisão cronológica treino/teste
# ---------------------------------------------------------------------
# IMPORTANTE:
# Não se usa train_test_split com shuffle=True, porque isso misturaria
# passado e futuro e criaria fuga de informação (data leakage).

split_idx = int(len(df_q1) * 0.8)

treino = df_q1.iloc[:split_idx].copy()
teste = df_q1.iloc[split_idx:].copy()

if treino.empty or teste.empty:
    raise ValueError("A divisão treino/teste gerou subconjuntos vazios.")

print(
    "Dados Q1 preparados com sucesso.\n"
    f"Registos de treino: {len(treino)}\n"
    f"Registos de teste: {len(teste)}"
)


# ---------------------------------------------------------------------
# 6. Modelo 1 — Baseline Naïve
# ---------------------------------------------------------------------
# O Baseline Naïve assume que o valor previsto para o dia atual é igual
# ao valor observado no dia anterior.
#
# Este modelo simples é essencial como linha de base. Um modelo complexo
# só acrescenta valor se superar esta referência mínima.

teste["pred_baseline"] = teste["lag_1"]


# ---------------------------------------------------------------------
# 7. Modelo 2 — ARIMA (1,1,1)
# ---------------------------------------------------------------------
# Modelo estatístico clássico para séries temporais.
#
# Nota:
# Aqui o ARIMA é ajustado apenas sobre o conjunto de treino e faz uma
# previsão multietapa para todo o horizonte de teste. Não é atualizado
# diariamente com novos valores reais.

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    historico = treino["new_cases_7_day_avg_right"].values

    modelo_arima = ARIMA(historico, order=(1, 1, 1))
    modelo_arima_fit = modelo_arima.fit()

    teste["pred_arima"] = modelo_arima_fit.forecast(steps=len(teste))


# ---------------------------------------------------------------------
# 8. Modelo 3 — Random Forest Regressor
# ---------------------------------------------------------------------
# Algoritmo de aprendizagem supervisionada baseado em árvores de decisão.
#
# Mantém-se a lógica da versão anterior:
# o modelo usa lags de 1, 7 e 14 dias como variáveis explicativas.

features = ["lag_1", "lag_7", "lag_14"]

X_train = treino[features]
y_train = treino["new_cases_7_day_avg_right"]

X_test = teste[features]

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

teste["pred_rf"] = rf_model.predict(X_test)


# ---------------------------------------------------------------------
# 9. Avaliação dos modelos
# ---------------------------------------------------------------------
# Calculam-se as métricas no conjunto de teste, comparando previsões
# com os valores reais observados.

y_true = teste["new_cases_7_day_avg_right"]

modelos = {
    "Baseline Naïve": "pred_baseline",
    "ARIMA (1,1,1)": "pred_arima",
    "Random Forest Regressor": "pred_rf",
}

metricas = {}

print("\n--- DESEMPENHO DOS MODELOS NO CONJUNTO DE TESTE — Q1 ---")

for nome_modelo, coluna_predicao in modelos.items():
    metricas[nome_modelo] = calcular_metricas(
        y_true,
        teste[coluna_predicao],
    )

    print(f"\n{nome_modelo}")
    print(f"MAE : {metricas[nome_modelo]['MAE']:.2f}")
    print(f"RMSE: {metricas[nome_modelo]['RMSE']:.2f}")
    print(f"R²  : {metricas[nome_modelo]['R2']:.4f}")


# ---------------------------------------------------------------------
# 10. Exportação do output analítico
# ---------------------------------------------------------------------
# O ficheiro outputs_q1.csv constitui a matriz final para:
# - análise da Questão 1;
# - geração de gráficos;
# - comparação entre valor real e valores previstos;
# - sustentação empírica do relatório final.

output_df = teste[
    [
        "date",
        "new_cases_7_day_avg_right",
        "pred_baseline",
        "pred_arima",
        "pred_rf",
    ]
].copy()

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

output_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nFicheiro Q1 exportado com sucesso para: {OUTPUT_FILE}")