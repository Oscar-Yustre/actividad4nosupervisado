# --- Script para generar el dataset sintético: generate_dataset.py ---

import pandas as pd
import numpy as np
import random

# Número de estaciones a simular
num_stations = 100

# Generar IDs de estación
station_ids = [f"EST{str(i).zfill(3)}" for i in range(1, num_stations + 1)]

# Generar datos sintéticos con cierta lógica simulada
data = {
    'station_id': station_ids,
    # Afluencia diaria promedio (valores entre 500 y 50000)
    'avg_daily_entries': np.random.randint(500, 50001, size=num_stations),
    # Proporción de uso en hora pico (más alto para estaciones con más afluencia)
    # Usamos una base aleatoria y añadimos un factor relacionado con la afluencia
    'peak_hour_ratio': np.clip(0.4 + np.random.normal(0, 0.15, size=num_stations) + (np.random.rand(num_stations) * 0.000005 * np.random.randint(500, 50001, size=num_stations)), 0.2, 0.9),
    # Proporción de uso en fin de semana (inversamente relacionado a hora pico, algunas estaciones pueden ser más turísticas/residenciales)
    'weekend_usage_ratio': np.clip(0.3 + np.random.normal(0, 0.1, size=num_stations) - (np.random.rand(num_stations) * 0.2 * (np.random.rand(num_stations))), 0.05, 0.5),
    # Número de líneas conectadas (más probable en estaciones concurridas)
    'num_connecting_lines': np.random.choice([1, 2, 3, 4, 5], size=num_stations, p=[0.4, 0.3, 0.15, 0.1, 0.05]) + np.random.randint(0, 3, size=num_stations) * (np.random.randint(500, 50001, size=num_stations) > 25000) # Añade más líneas si la estación es muy concurrida
}
# Asegurar que el número de líneas no sea 0 o negativo
data['num_connecting_lines'] = np.clip(data['num_connecting_lines'], 1, None).astype(int)


# Crear DataFrame
df_stations = pd.DataFrame(data)

# Redondear ratios para mayor legibilidad
df_stations['peak_hour_ratio'] = df_stations['peak_hour_ratio'].round(3)
df_stations['weekend_usage_ratio'] = df_stations['weekend_usage_ratio'].round(3)

# Guardar en CSV
file_path = 'stations_data.csv'
df_stations.to_csv(file_path, index=False)

print(f"Dataset sintético guardado en: {file_path}")
print("Primeras 5 filas:")
print(df_stations.head())

# --- Fin del script generate_dataset.py ---