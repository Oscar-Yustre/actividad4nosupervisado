# --- Script de Análisis de Clustering: clustering_analysis.py ---

# 1. Importar Librerías Necesarias
import pandas as pd                # Para manipulación de datos (DataFrames)
import numpy as np                 # Para operaciones numéricas
from sklearn.cluster import KMeans # Algoritmo de Clustering K-Means
from sklearn.preprocessing import StandardScaler # Para escalar los datos (importante para K-Means)
import matplotlib.pyplot as plt    # Para crear gráficos
import seaborn as sns              # Para gráficos más estéticos y complejos

print("Librerías importadas correctamente.")

# 2. Cargar los Datos
# Asegúrate de que el archivo 'stations_data.csv' esté en el mismo directorio
# o proporciona la ruta completa al archivo.
try:
    data_path = 'stations_data.csv'
    df = pd.read_csv(data_path)
    print(f"\nDatos cargados desde '{data_path}'. Dimensiones: {df.shape}")
    print("Primeras filas del dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: El archivo '{data_path}' no se encontró.")
    print("Asegúrate de generar primero el dataset ejecutando 'generate_dataset.py' o colocando el CSV en la ruta correcta.")
    exit() # Termina el script si no encuentra el archivo

# 3. Preprocesamiento de Datos
print("\n--- Preprocesamiento ---")
# Seleccionar las características numéricas que usaremos para el clustering.
# Excluimos 'station_id' porque es un identificador, no una característica medible para agrupar.
features = ['avg_daily_entries', 'peak_hour_ratio', 'weekend_usage_ratio', 'num_connecting_lines']
X = df[features]

print(f"Características seleccionadas para clustering: {features}")

# Escalar los datos: K-Means es sensible a la escala de las características.
# Usamos StandardScaler para que cada característica tenga media 0 y desviación estándar 1.
scaler = StandardScaler() # Crear una instancia del escalador
X_scaled = scaler.fit_transform(X) # Ajustar y transformar los datos

print("Datos escalados (primeras 5 filas):")
print(X_scaled[:5])

# 4. Determinar el Número Óptimo de Clústeres (k) usando el Método del Codo (Elbow Method)
print("\n--- Determinando el número óptimo de clústeres (k) ---")
inertia = [] # Lista para guardar la inercia (suma de cuadrados intra-clúster) para cada k
k_range = range(1, 11) # Probaremos k desde 1 hasta 10

for k in k_range:
    kmeans = KMeans(n_clusters=k,  # Número de clústeres a probar
                    init='k-means++', # Método de inicialización (mejora la convergencia)
                    n_init=10,       # Número de veces que se ejecutará K-Means con diferentes centroides iniciales
                    max_iter=300,    # Número máximo de iteraciones por ejecución
                    random_state=42) # Fija la semilla aleatoria para reproducibilidad
    kmeans.fit(X_scaled) # Ajustar el modelo a los datos escalados
    inertia.append(kmeans.inertia_) # Guardar la inercia del modelo ajustado

# Graficar el Método del Codo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Inercia (Suma de cuadrados intra-clúster)')
plt.title('Método del Codo para determinar k óptimo')
plt.xticks(k_range)
plt.grid(True)
plt.show() # Muestra el gráfico

# Nota: Busca el "codo" en el gráfico, el punto donde la tasa de disminución de la inercia se reduce significativamente.
# Este punto sugiere un buen equilibrio entre el número de clústeres y la cohesión dentro de ellos.
# Basado en el gráfico, deberás elegir un valor para 'optimal_k'. ¡Inspecciona el gráfico generado!
# Por ejemplo, si el codo parece estar en k=3 o k=4, elige ese valor.
# Para este ejemplo, asumiremos que el codo sugiere k=4 (¡ajústalo según tu gráfico!).
optimal_k = 4
print(f"Basado en el gráfico del codo (inspección visual), elegimos k = {optimal_k}")

# 5. Aplicar K-Means con el Número Óptimo de Clústeres
print(f"\n--- Aplicando K-Means con k = {optimal_k} ---")
kmeans_final = KMeans(n_clusters=optimal_k,
                      init='k-means++',
                      n_init=10,
                      max_iter=300,
                      random_state=42)
kmeans_final.fit(X_scaled) # Ajustar el modelo final

# Obtener las etiquetas de clúster para cada estación
cluster_labels = kmeans_final.labels_

# Añadir las etiquetas de clúster al DataFrame original para análisis
df['cluster'] = cluster_labels
print("Etiquetas de clúster asignadas (primeras filas con la nueva columna 'cluster'):")
print(df.head())

# 6. Analizar y Visualizar los Resultados
print("\n--- Análisis de los Clústeres ---")
# Calcular las características promedio de cada clúster
cluster_summary = df.groupby('cluster')[features].mean()
print("Características promedio por clúster:")
print(cluster_summary)

# Visualizar los clústeres
# Usaremos pairplot para ver las relaciones entre pares de variables, coloreadas por clúster.
# Esto ayuda a entender cómo se separan los grupos en el espacio de características.
print("\nGenerando visualización de clústeres (pairplot)...")
# Añadimos 'cluster' temporalmente a las features para el pairplot y luego lo quitamos si es necesario
# Creamos una copia para no modificar el df original innecesariamente en esta visualización
df_plot = df.copy()
sns.pairplot(df_plot, vars=features, hue='cluster', palette='viridis', diag_kind='kde')
plt.suptitle(f'Visualización de Clústeres (k={optimal_k})', y=1.02) # Título general por encima de los gráficos
plt.show()

# También podemos hacer scatter plots específicos si el pairplot es muy grande
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='avg_daily_entries', y='peak_hour_ratio', hue='cluster', palette='viridis', s=100, alpha=0.7)
plt.title(f'Clusters: Afluencia Diaria vs Proporción Hora Pico (k={optimal_k})')
plt.xlabel('Entradas Diarias Promedio')
plt.ylabel('Proporción Hora Pico')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='avg_daily_entries', y='num_connecting_lines', hue='cluster', palette='viridis', s=100, alpha=0.7)
plt.title(f'Clusters: Afluencia Diaria vs Líneas Conectadas (k={optimal_k})')
plt.xlabel('Entradas Diarias Promedio')
plt.ylabel('Número de Líneas Conectadas')
plt.grid(True)
plt.show()

print("\n--- Interpretación de los Clústeres (Ejemplo basado en promedios) ---")
# Esta parte es CUALITATIVA y depende de los resultados específicos que obtengas.
# Observa la tabla 'cluster_summary' y los gráficos para describir cada clúster.
# Ejemplo de interpretación (¡DEBES AJUSTAR ESTO A TUS RESULTADOS!):
# Clúster 0: Podrían ser estaciones con baja afluencia, pocas líneas, quizás más activas fuera de hora pico (residenciales?).
# Clúster 1: Podrían ser estaciones muy concurridas, con muchas líneas, muy activas en hora pico (hubs centrales?).
# Clúster 2: Estaciones con afluencia media, quizás con alto uso en fines de semana (turísticas/ocio?).
# Clúster 3: Otro perfil intermedio, quizás estaciones de transferencia importantes pero no las más grandes.

# 7. Guardar los resultados (opcional)
# Podrías guardar el DataFrame con los clústeres asignados
results_path = 'stations_clustered.csv'
df.to_csv(results_path, index=False)
print(f"\nDataFrame con clústeres asignados guardado en: {results_path}")

print("\n--- Fin del Análisis ---")

# --- Fin del script clustering_analysis.py ---