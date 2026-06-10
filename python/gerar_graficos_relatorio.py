import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Detetar caminhos absolutos automaticamente para evitar erros
script_dir = os.path.dirname(os.path.abspath(__file__)) # Deteta C:\bi\python
base_dir = os.path.dirname(script_dir)                  # Sobe para C:\bi
datasets_dir = os.path.join(base_dir, 'datasets')       # Aponta para C:\bi\datasets
out_dir = os.path.join(base_dir, 'graficos')            # Aponta para C:\bi\graficos

# Criar pasta para guardar as imagens dos gráficos
os.makedirs(out_dir, exist_ok=True)

# 2. Gráfico Q1: Previsão de Incidência
df_q1 = pd.read_csv(os.path.join(datasets_dir, 'outputs_q1.csv'))
df_q1['date'] = pd.to_datetime(df_q1['date'])

plt.figure(figsize=(12, 6))
plt.plot(df_q1['date'], df_q1['new_cases_7_day_avg_right'], label='Real (Média 7 Dias)', color='black', linewidth=2)
plt.plot(df_q1['date'], df_q1['pred_baseline'], label='Baseline Naïve', linestyle='--')
plt.plot(df_q1['date'], df_q1['pred_rf'], label='Random Forest', linestyle='-.')
plt.title('Q1 - Previsão de Incidência em Portugal')
plt.xlabel('Data')
plt.ylabel('Novos Casos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'grafico_q1_incidencia.png'), bbox_inches='tight')
plt.close()

# 3. Gráfico Q2: Mortalidade Acumulada
df_q2 = pd.read_csv(os.path.join(datasets_dir, 'outputs_q2.csv'))
df_q2['date'] = pd.to_datetime(df_q2['date'])

plt.figure(figsize=(12, 6))
plt.plot(df_q2['date'], df_q2['total_deaths_per_million'], label='Óbitos Reais (Acumulado)', color='black', linewidth=2)
plt.plot(df_q2['date'], df_q2['pred_linear'], label='Regressão Linear', linestyle='--')
plt.plot(df_q2['date'], df_q2['pred_polynomial'], label='Regressão Polinomial', linestyle='-.')
plt.title('Q2 - Falha dos Modelos na Mortalidade Acumulada')
plt.xlabel('Data')
plt.ylabel('Óbitos por Milhão')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'grafico_q2_mortalidade.png'), bbox_inches='tight')
plt.close()

# 4. Gráfico Q3: Estabilização da CFR
df_q3 = pd.read_csv(os.path.join(datasets_dir, 'outputs_q3.csv'))
df_q3['date'] = pd.to_datetime(df_q3['date'])

plt.figure(figsize=(12, 6))
plt.fill_between(df_q3['date'], df_q3['cfr_mundo'], label='CFR Mundial', color='skyblue', alpha=0.4)
plt.plot(df_q3['date'], df_q3['cfr_portugal'], label='CFR Portugal', color='darkblue', linewidth=2)
plt.title('Q3 - Estabilização da Taxa de Letalidade (CFR) após 1.000 Casos')
plt.xlabel('Data')
plt.ylabel('Taxa de Letalidade (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'grafico_q3_cfr.png'), bbox_inches='tight')
plt.close()

print(f"Gráficos gerados com sucesso na pasta: {out_dir}")