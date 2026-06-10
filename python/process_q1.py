import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

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

# Isolar o indicador alvo da Questão 1 (Média móvel de 7 dias)
df_q1 = df_pt[['date', 'new_cases_7_day_avg_right']].dropna().copy()

# 2. Criação de Variáveis de Atraso (Lag Features) para o Random Forest
df_q1['lag_1'] = df_q1['new_cases_7_day_avg_right'].shift(1)
df_q1['lag_7'] = df_q1['new_cases_7_day_avg_right'].shift(7)
df_q1['lag_14'] = df_q1['new_cases_7_day_avg_right'].shift(14)
df_q1 = df_q1.dropna().reset_index(drop=True)

# 3. Divisão Sequencial Temporal (80% Treino / 20% Teste)
split_idx = int(len(df_q1) * 0.8)
treino = df_q1.iloc[:split_idx].copy()
teste = df_q1.iloc[split_idx:].copy()

print(f"Dados Q1 prontos. Registos de Treino: {len(treino)} | Registos de Teste: {len(teste)}")

# --- MODELO 1: BASELINE NAÏVE ---
teste['pred_baseline'] = teste['lag_1']

# --- MODELO 2: MODELO ESTATÍSTICO ARIMA ---
history = list(treino['new_cases_7_day_avg_right'].values)
model_arima = ARIMA(history, order=(1, 1, 1))
model_arima_fit = model_arima.fit()
teste['pred_arima'] = model_arima_fit.forecast(steps=len(teste))

# --- MODELO 3: RANDOM FOREST REGRESSOR ---
X_train = treino[['lag_1', 'lag_7', 'lag_14']]
y_train = treino['new_cases_7_day_avg_right']
X_test = teste[['lag_1', 'lag_7', 'lag_14']]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
teste['pred_rf'] = rf_model.predict(X_test)

# --- AVALIAÇÃO DE MÉTRICAS MATEMÁTICAS ---
y_true = teste['new_cases_7_day_avg_right']
metrics = {}

for col in ['pred_baseline', 'pred_arima', 'pred_rf']:
    mae = mean_absolute_error(y_true, teste[col])
    rmse = np.sqrt(mean_squared_error(y_true, teste[col]))
    r2 = r2_score(y_true, teste[col])
    metrics[col] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

print("\n--- DESEMPENHO DOS MODELOS NO CONJUNTO DE TESTE (Q1) ---")
for model_name, m in metrics.items():
    print(f"{model_name.upper()}:")
    print(f"  MAE:  {m['MAE']:.2f} | RMSE: {m['RMSE']:.2f} | R²: {m['R2']:.4f}")

# --- EXPORTAÇÃO ---
output_df = teste[['date', 'new_cases_7_day_avg_right', 'pred_baseline', 'pred_arima', 'pred_rf']]
output_path = os.path.join('..', 'datasets', 'outputs_q1.csv')
output_df.to_csv(output_path, index=False)
print(f"\nFicheiro Q1 exportado para: {output_path}")