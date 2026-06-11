"""
gerar_graficos_relatorio.py

Geração automática dos gráficos usados no relatório final.

OBJETIVO
--------
Ler os outputs analíticos produzidos pelos scripts das Questões 1, 2, 3 e 4
e gerar gráficos em formato PNG para inclusão no relatório.

FICHEIROS DE ENTRADA
--------------------
datasets/outputs_q1.csv
datasets/outputs_q2.csv
datasets/outputs_q3.csv
datasets/outputs_q4.csv

FICHEIROS DE SAÍDA
------------------
graficos/grafico_q1_incidencia.png
graficos/grafico_q2_mortalidade.png
graficos/grafico_q3_cfr.png
graficos/grafico_q4_metricas.png

ALTERAÇÕES FACE À VERSÃO ANTERIOR
---------------------------------
1. Caminhos robustos com Path(__file__).
2. Validação explícita dos ficheiros de entrada.
3. Validação das colunas necessárias em cada output.
4. Compatibilidade com os nomes das colunas dos scripts revistos.
5. Remoção de referências a Qlik.
6. Geração de gráficos exclusivamente em Python.
7. Exportação em alta resolução para integração no relatório final.

NOTA
----
Os gráficos não substituem a análise crítica. Servem como suporte visual
para interpretar o comportamento dos modelos, os erros de previsão e a
comparação entre valores reais e estimados.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Definição robusta de caminhos
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASETS_DIR = BASE_DIR / "datasets"
GRAFICOS_DIR = BASE_DIR / "graficos"

Q1_FILE = DATASETS_DIR / "outputs_q1.csv"
Q2_FILE = DATASETS_DIR / "outputs_q2.csv"
Q3_FILE = DATASETS_DIR / "outputs_q3.csv"
Q4_FILE = DATASETS_DIR / "outputs_q4.csv"

GRAFICO_Q1 = GRAFICOS_DIR / "grafico_q1_incidencia.png"
GRAFICO_Q2 = GRAFICOS_DIR / "grafico_q2_mortalidade.png"
GRAFICO_Q3 = GRAFICOS_DIR / "grafico_q3_cfr.png"
GRAFICO_Q4 = GRAFICOS_DIR / "grafico_q4_metricas.png"


# ---------------------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------------------

def validar_ficheiro(caminho):
    """
    Confirma se o ficheiro existe antes da leitura.
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {caminho}\n"
            "Execute primeiro os scripts process_q1.py, process_q2.py, "
            "process_q3.py e process_q4.py."
        )


def normalizar_colunas(df):
    """
    Normaliza os nomes das colunas para minúsculas e underscores.

    Isto torna o script mais tolerante a pequenas diferenças formais
    nos ficheiros CSV.
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
    Verifica se o DataFrame contém todas as colunas necessárias.
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


def preparar_datas(df):
    """
    Converte a coluna date para datetime e ordena cronologicamente.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df


def guardar_grafico(caminho):
    """
    Guarda o gráfico com qualidade adequada para relatório.
    """
    plt.tight_layout()
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# 3. Preparação da pasta de gráficos
# ---------------------------------------------------------------------

GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 4. Gráfico Q1 — Incidência real vs previsões
# ---------------------------------------------------------------------

def gerar_grafico_q1():
    """
    Gera o gráfico comparativo da Questão 1.

    Mostra:
    - curva real da média móvel de novos casos;
    - previsão Baseline Naïve;
    - previsão ARIMA;
    - previsão Random Forest.
    """
    validar_ficheiro(Q1_FILE)

    df = pd.read_csv(Q1_FILE)
    df = normalizar_colunas(df)

    validar_colunas(
        df,
        [
            "date",
            "new_cases_7_day_avg_right",
            "pred_baseline",
            "pred_arima",
            "pred_rf",
        ],
        "outputs_q1.csv",
    )

    df = preparar_datas(df)

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["new_cases_7_day_avg_right"],
        label="Real observado",
        linewidth=2,
    )

    plt.plot(
        df["date"],
        df["pred_baseline"],
        label="Baseline Naïve",
        linestyle="--",
    )

    plt.plot(
        df["date"],
        df["pred_arima"],
        label="ARIMA (1,1,1)",
        linestyle=":",
    )

    plt.plot(
        df["date"],
        df["pred_rf"],
        label="Random Forest",
        linestyle="-.",
    )

    plt.title(
        "Q1 — Incidência diária suavizada em Portugal: real vs previsões"
    )
    plt.xlabel("Data")
    plt.ylabel("Novos casos — média móvel de 7 dias")
    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_grafico(GRAFICO_Q1)

    print(f"Gráfico Q1 gerado: {GRAFICO_Q1}")


# ---------------------------------------------------------------------
# 5. Gráfico Q2 — Mortalidade real vs previsões
# ---------------------------------------------------------------------

def gerar_grafico_q2():
    """
    Gera o gráfico comparativo da Questão 2.

    Mostra:
    - mortalidade acumulada real por milhão;
    - previsão da Regressão Linear Múltipla;
    - previsão da Regressão Polinomial Grau 2.
    """
    validar_ficheiro(Q2_FILE)

    df = pd.read_csv(Q2_FILE)
    df = normalizar_colunas(df)

    validar_colunas(
        df,
        [
            "date",
            "total_deaths_per_million",
            "pred_regressao_linear",
            "pred_regressao_polinomial_g2",
        ],
        "outputs_q2.csv",
    )

    df = preparar_datas(df)

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["total_deaths_per_million"],
        label="Real observado",
        linewidth=2,
    )

    plt.plot(
        df["date"],
        df["pred_regressao_linear"],
        label="Regressão Linear Múltipla",
        linestyle="--",
    )

    plt.plot(
        df["date"],
        df["pred_regressao_polinomial_g2"],
        label="Regressão Polinomial Grau 2",
        linestyle="-.",
    )

    plt.title(
        "Q2 — Mortalidade acumulada por milhão em Portugal: real vs previsões"
    )
    plt.xlabel("Data")
    plt.ylabel("Óbitos acumulados por milhão")
    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_grafico(GRAFICO_Q2)

    print(f"Gráfico Q2 gerado: {GRAFICO_Q2}")


# ---------------------------------------------------------------------
# 6. Gráfico Q3 — CFR Portugal vs Mundo
# ---------------------------------------------------------------------

def gerar_grafico_q3():
    """
    Gera o gráfico comparativo da Taxa de Letalidade.

    Mostra:
    - CFR de Portugal;
    - CFR da referência mundial.
    """
    validar_ficheiro(Q3_FILE)

    df = pd.read_csv(Q3_FILE)
    df = normalizar_colunas(df)

    validar_colunas(
        df,
        [
            "date",
            "cfr_portugal",
            "cfr_mundo",
        ],
        "outputs_q3.csv",
    )

    df = preparar_datas(df)

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["cfr_portugal"],
        label="Portugal",
        linewidth=2,
    )

    plt.plot(
        df["date"],
        df["cfr_mundo"],
        label="Mundo",
        linestyle="--",
    )

    plt.title(
        "Q3 — Evolução da Taxa de Letalidade (CFR): Portugal vs Mundo"
    )
    plt.xlabel("Data")
    plt.ylabel("CFR (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    guardar_grafico(GRAFICO_Q3)

    print(f"Gráfico Q3 gerado: {GRAFICO_Q3}")


# ---------------------------------------------------------------------
# 7. Gráfico Q4 — Métricas de validação
# ---------------------------------------------------------------------

def gerar_grafico_q4():
    """
    Gera um gráfico de barras com o RMSE dos modelos.

    O RMSE foi escolhido para representação visual porque penaliza mais
    fortemente erros elevados, sendo útil para comparar a estabilidade
    dos modelos.

    A tabela completa de métricas continua disponível em outputs_q4.csv.
    """
    validar_ficheiro(Q4_FILE)

    df = pd.read_csv(Q4_FILE)
    df = normalizar_colunas(df)

    validar_colunas(
        df,
        [
            "questao",
            "modelo",
            "mae",
            "rmse",
            "r2",
        ],
        "outputs_q4.csv",
    )

    # Cria uma etiqueta curta para facilitar a leitura no eixo X.
    df["modelo_curto"] = df["questao"] + "\n" + df["modelo"]

    plt.figure(figsize=(12, 6))

    plt.bar(
        df["modelo_curto"],
        df["rmse"],
    )

    plt.title("Q4 — Comparação do RMSE entre modelos")
    plt.xlabel("Modelo")
    plt.ylabel("RMSE")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    guardar_grafico(GRAFICO_Q4)

    print(f"Gráfico Q4 gerado: {GRAFICO_Q4}")


# ---------------------------------------------------------------------
# 8. Execução principal
# ---------------------------------------------------------------------

def main():
    """
    Executa a geração de todos os gráficos.
    """
    gerar_grafico_q1()
    gerar_grafico_q2()
    gerar_grafico_q3()
    gerar_grafico_q4()

    print("\nTodos os gráficos foram gerados com sucesso.")


if __name__ == "__main__":
    main()