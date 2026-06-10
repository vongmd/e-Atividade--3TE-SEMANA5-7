import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Definir os caminhos dos ficheiros de previsão gerados
q1_path = os.path.join('..', 'datasets', 'outputs_q1.csv')
q2_path = os.path.join('..', 'datasets', 'outputs_q2.csv')

if not os.path.exists(q1_path) or not os.path.exists(q2_path):
    raise FileNotFoundError("Por favor, execute primeiro os scripts process_q1.py e process_q2.py para gerar as previsões.")

# 2. Ler os outputs das séries temporais preditivas
df_q1 = pd.read_csv(q1_path)
df_q2 = pd.read_csv(q2_path)

metrics_summary = []

# 3. Processar Balanço Estatístico da Questão 1 (Incidência)
y_true_q1 = df_q1['new_cases_7_day_avg_right']
models_q1 = [
    ('pred_baseline', 'Baseline Naïve'),
    ('pred_arima', 'ARIMA (1,1,1)'),
    ('pred_rf', 'Random Forest')
]

for col, name in models_q1:
    mae = mean_absolute_error(y_true_q1, df_q1[col])
    rmse = np.sqrt(mean_squared_error(y_true_q1, df_q1[col]))
    r2 = r2_score(y_true_q1, df_q1[col])
    metrics_summary.append({
        'Questao': 'Q1 - Incidência',
        'Modelo_Baseline': name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    })

# 4. Processar Balanço Estatístico da Questão 2 (Mortalidade Risco)
y_true_q2 = df_q2['total_deaths_per_million']
models_q2 = [
    ('pred_linear', 'Regressão Linear Múltipla'),
    ('pred_polynomial', 'Regressão Polinomial G2')
]

for col, name in models_q2:
    mae = mean_absolute_error(y_true_q2, df_q2[col])
    rmse = np.sqrt(mean_squared_error(y_true_q2, df_q2[col]))
    r2 = r2_score(y_true_q2, df_q2[col])
    metrics_summary.append({
        'Questao': 'Q2 - Mortalidade',
        'Modelo_Baseline': name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    })

# 5. Criar DataFrame Consolidado de Auditoria (Ground Truth)
df_q4 = pd.DataFrame(metrics_summary)

print("\n--- BALANÇO ESTATÍSTICO CONSOLIDADO DE VALIDAÇÃO (Q4) ---")
print(df_q4.to_string(index=False))

# 6. Exportação do Output de Auditoria para o Qlik Sense (Tabelas de KPI)
output_path = os.path.join('..', 'datasets', 'outputs_q4.csv')
df_q4.to_csv(output_path, index=False)
print(f"\nFicheiro de integração Q4 exportado com sucesso para: {output_path}")