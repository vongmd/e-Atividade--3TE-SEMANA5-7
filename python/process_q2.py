import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. Carregar os dados limpos de Portugal
csv_path = os.path.join('..', 'datasets', 'covid_paises_limpo.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Ficheiro não encontrado em: {csv_path}")

df = pd.read_csv(csv_path)

# Normalização de colunas (letras minúsculas e underscores)
df.columns = df.columns.str.lower().str.replace(' ', '_')
df['date'] = pd.to_datetime(df['date'])

# CORREÇÃO: Filtragem robusta que ignora maiúsculas/minúsculas nos dados
df_pt = df[df['country'].str.lower() == 'portugal'].sort_values('date').copy()

# 2. Engenharia de Características (Features)
# CORREÇÃO: Alterado 'total_cases_per_million' para 'total_cases' conforme a Tabela 2 do relatório
df_q2 = df_pt[['date', 'total_deaths_per_million', 'total_cases', 'new_cases_7_day_avg_right']].copy()
df_q2['lag_14_new_cases'] = df_q2['new_cases_7_day_avg_right'].shift(14)
df_q2['lag_21_new_cases'] = df_q2['new_cases_7_day_avg_right'].shift(21)
df_q2 = df_q2.dropna().reset_index(drop=True)

# 3. Divisão Sequencial Temporal (80% Treino / 20% Teste)
split_idx = int(len(df_q2) * 0.8)
treino = df_q2.iloc[:split_idx].copy()
teste = df_q2.iloc[split_idx:].copy()

print(f"Dados Q2 prontos. Registos de Treino: {len(treino)} | Registos de Teste: {len(teste)}")

# CORREÇÃO: Ajuste do nome da feature para 'total_cases'
features = ['total_cases', 'lag_14_new_cases', 'lag_21_new_cases']
X_train = treino[features]
y_train = treino['total_deaths_per_million']
X_test = teste[features]
y_test = teste['total_deaths_per_million']

# --- MODELO 1: REGRESSÃO LINEAR MÚLTIPLA ---
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
teste['pred_linear'] = linear_model.predict(X_test)

# --- MODELO 2: REGRESSÃO POLINOMIAL (GRAU 2) ---
poly_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
poly_model.fit(X_train, y_train)
teste['pred_polynomial'] = poly_model.predict(X_test)

# --- AVALIAÇÃO DE MÉTRICAS MATEMÁTICAS ---
metrics_q2 = {}
for col in ['pred_linear', 'pred_polynomial']:
    mae = mean_absolute_error(y_test, teste[col])
    rmse = np.sqrt(mean_squared_error(y_test, teste[col]))
    r2 = r2_score(y_test, teste[col])
    metrics_q2[col] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

print("\n--- DESEMPENHO DOS MODELOS DE MONITORIZAÇÃO DE RISCO (Q2) ---")
for model_name, m in metrics_q2.items():
    print(f"{model_name.upper()}:")
    print(f"  MAE:  {m['MAE']:.2f} | RMSE: {m['RMSE']:.2f} | R²: {m['R2']:.4f}")

# --- EXPORTAÇÃO ---
output_df = teste[['date', 'total_deaths_per_million', 'pred_linear', 'pred_polynomial']]
output_path = os.path.join('..', 'datasets', 'outputs_q2.csv')
output_df.to_csv(output_path, index=False)
print(f"\nFicheiro Q2 exportado para: {output_path}")