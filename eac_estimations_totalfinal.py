import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

sns.set(style="whitegrid")


def predecir_EAC(df, site, fecha_objetivo):
    metrics = None  # Definir metrics al inicio para evitar el error UnboundLocalError

    if site == 'Total':
        # Crear la columna 'EAC' en el DataFrame general (para todos los sites)
        df['ACWP'] = pd.to_numeric(df['ACWP'].replace({',': ''}, regex=True), errors='coerce')
        df['ETC'] = pd.to_numeric(df['ETC'].replace({',': ''}, regex=True), errors='coerce')
        df['ACWP'] = df['ACWP'].fillna(0)
        df['ETC'] = df['ETC'].fillna(0)
        df['EAC'] = df['ACWP'] + df['ETC']
        
        # Agrupar por fecha y sumar los EAC de todos los sites
        df_total = df.groupby('DATE').agg({'EAC': 'sum'}).reset_index()
        df_total['DATE'] = pd.to_datetime(df_total['DATE'], format='%b-%y', errors='coerce')
        df_total = df_total.dropna(subset=['DATE']).sort_values('DATE')
        df_total['DAYS'] = (df_total['DATE'] - df_total['DATE'].min()).dt.days
        
        # Definir X y y correctamente para el Total
        X = df_total[['DAYS']]
        y = df_total['EAC']
        title = 'Predicción de EAC para el Total de Sites'

        df_site = df_total  # Asignamos df_total a df_site para que sea coherente
    else:
        # Filtrar por un solo site
        df_site = df[df['SITE'].str.strip() == site].copy()
        if df_site.empty:
            print(f"No se encontraron datos para el site '{site}'.")
            return None, None

        df_site['DATE'] = pd.to_datetime(df_site['DATE'].str.strip(), format='%b-%y', errors='coerce')
        df_site = df_site.dropna(subset=['DATE']).sort_values('DATE')

        for col in ['ACWP', 'ETC']:
            df_site[col] = pd.to_numeric(df_site[col].replace({',': ''}, regex=True), errors='coerce')

        df_site['ACWP'] = df_site['ACWP'].fillna(0)
        df_site['ETC'] = df_site['ETC'].fillna(0)
        df_site['EAC'] = df_site['ACWP'] + df_site['ETC']
        df_site = df_site[~((df_site['ACWP'] == 0) & (df_site['ETC'] == 0))]

        if len(df_site) < 4:
            print(f"No hay suficientes datos históricos para el site '{site}'.")
            return None, None

        # Comprobar si el último registro tiene ETC = 0
        last_record = df_site.iloc[-1]
        if last_record['ETC'] <= 1e-6:
            print(f"El site '{site}' ya ha terminado; la predicción de EAC será igual al último valor registrado.")
            eac_estimado = last_record['EAC']
            # Establecer metrics a None para indicar que no se generaron métricas
            metrics = {'MSE': None, 'R2': None, 'Intervalos': None}
            return eac_estimado, metrics

        title = f'Predicción de EAC para el site {site}'

        # Definir X y y para el site individual
        df_site['DAYS'] = (df_site['DATE'] - df_site['DATE'].min()).dt.days
        X = df_site[['DAYS']]
        y = df_site['EAC']

    # Modelado con RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    fecha_objetivo_dt = pd.to_datetime(fecha_objetivo, format='%Y-%m-%d', errors='coerce')
    t_objetivo = (fecha_objetivo_dt - df_site['DATE'].min()).days

    eac_estimado = model.predict([[t_objetivo]])[0]

    y_pred_train = model.predict(X)
    mse = mean_squared_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)

    std_dev = np.std(y - y_pred_train)
    percentiles = [0.01, 0.025, 0.05, 0.1]
    intervalos = {
        f'{int((1 - p) * 100)}%': (eac_estimado - norm.ppf(1 - p) * std_dev, eac_estimado + norm.ppf(1 - p) * std_dev)
        for p in percentiles
    }

    # Asignar las métricas al diccionario metrics
    metrics = {'MSE': mse, 'R2': r2, 'Intervalos': intervalos}

    # Mostrar los resultados del modelo
    print(f"\nResultados del modelo para el site '{site}' (Fecha objetivo: {fecha_objetivo}):")
    print(f"Valor puntual estimado de EAC: {eac_estimado:.2f}")
    print(f"Métricas de ajuste del modelo:")

    # Comprobar si las métricas son None antes de imprimir
    if metrics['MSE'] is not None and metrics['R2'] is not None:
        print(f"MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.2f}")
    else:
        print("MSE y R² no disponibles debido a que el sitio ha terminado.")
        
    print("Intervalos de confianza:")
    if metrics['Intervalos'] is not None:
        for nivel, rango in metrics['Intervalos'].items():
            print(f"{nivel}: [{rango[0]:.2f}, {rango[1]:.2f}]")
    else:
        print("No se generaron intervalos de confianza ya que el sitio ha terminado.")

    # Gráfico
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df_site['DATE'], y=df_site['EAC'], marker='o', label='Histórico EAC')
    plt.axvline(x=fecha_objetivo_dt, color='gray', linestyle='--', label='Fecha Objetivo')
    plt.scatter(fecha_objetivo_dt, eac_estimado, color='red', label='Predicción puntual')

    # Añadir los intervalos de confianza en la gráfica
    if metrics['Intervalos'] is not None:
        for nivel, (low, high) in intervalos.items():
            plt.fill_between([fecha_objetivo_dt], low, high, alpha=0.3, label=f'Intervalo {nivel}')
    
    # Añadir R² a la gráfica
    if metrics['MSE'] is not None and metrics['R2'] is not None:
        plt.text(0.95, 0.95, f'R²: {r2:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='green')
        # Añadir MSE a la gráfica
        plt.text(0.95, 0.90, f'MSE: {mse:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='green')

    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('EAC')
    plt.legend()
    plt.show()

    return eac_estimado, metrics


# --- Programa principal ---
df = pd.read_csv('EAC_data_base.csv')

# Opción para elegir un site o todos los sites
site_opcion = input("Ingresa el nombre del site o 'Total' para todos los sites: ").strip()
fecha_objetivo = input("Ingresa la fecha de predicción (YYYY-MM-DD): ").strip()

eac_estimado, metrics = predecir_EAC(df, site_opcion, fecha_objetivo)

if eac_estimado is not None:
    print(f"\nPredicción para el site '{site_opcion}' en fecha {fecha_objetivo}:")
    print(f"Valor puntual estimado: {eac_estimado:.2f}")
    if metrics:
        print("Métricas de ajuste:")
        if metrics['MSE'] is not None and metrics['R2'] is not None:
            print(f"MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.2f}")
        else:
            print("MSE y R² no disponibles debido a que el sitio ha terminado.")
        print("Intervalos de confianza:")
        if metrics['Intervalos'] is not None:
            for nivel, rango in metrics['Intervalos'].items():
                print(f"{nivel}: [{rango[0]:.2f}, {rango[1]:.2f}]")
        else:
            print("No se generaron intervalos de confianza ya que el sitio ha terminado.")
else:
    print("No se pudo realizar la predicción.")
