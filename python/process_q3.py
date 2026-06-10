import os
import pandas as pd

# 1. Definir os caminhos de leitura dos datasets limpos da etapa anterior
paises_path = os.path.join('..', 'datasets', 'covid_paises_limpo.csv')
global_path = os.path.join('..', 'datasets', 'covid_global_limpo.csv')

if not os.path.exists(paises_path) or not os.path.exists(global_path):
    raise FileNotFoundError("Certifique-se de que os ficheiros limpos existem em datasets/")

# 2. Carregar e normalizar colunas de Portugal
df_paises = pd.read_csv(paises_path)
df_paises.columns = df_paises.columns.str.lower().str.replace(' ', '_')
df_pt = df_paises[df_paises['country'].str.lower() == 'portugal'].copy()

# 3. Carregar e normalizar colunas da Referência Mundial
df_global = pd.read_csv(global_path)
df_global.columns = df_global.columns.str.lower().str.replace(' ', '_')

# 4. APLICAÇÃO DO FILTRO DE ESTABILIZAÇÃO (Relatório Secção 2.3)
# Bane os registos iniciais com volume acumulado inferior a 1.000 casos para mitigar ruído matemático
df_pt_estab = df_pt[df_pt['total_cases'] >= 1000].copy()
df_wd_estab = df_global[df_global['total_cases'] >= 1000].copy()

# 5. Isolar as colunas de interesse (CFR) e renomear para a integração
df_pt_cfr = df_pt_estab[['date', 'cfr']].rename(columns={'cfr': 'cfr_portugal'})
df_wd_cfr = df_wd_estab[['date', 'cfr']].rename(columns={'cfr': 'cfr_mundo'})

# 6. Combinar as séries temporais numa única matriz por data
df_q3 = pd.merge(df_pt_cfr, df_wd_cfr, on='date', how='inner')
df_q3['date'] = pd.to_datetime(df_q3['date'])
df_q3 = df_q3.sort_values('date')

print(f"Dados Q3 processados. Registos após estabilização (>=1000 casos): {len(df_q3)}")

# 7. Exportação do Output para o Qlik Sense (Gráfico de Área Comparativo)
output_path = os.path.join('..', 'datasets', 'outputs_q3.csv')
df_q3.to_csv(output_path, index=False)
print(f"Ficheiro de integração Q3 exportado com sucesso para: {output_path}")