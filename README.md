# actividad4nosupervisado
#oscar david yustre trujillo
#matria inteligencia artificial

Objetivo del Análisis: Agrupar estaciones de transporte masivo según características operativas usando K-Means.
Preprocesamiento Realizado: Se seleccionaron las características numéricas y se escalaron usando StandardScaler para asegurar que todas tuvieran el mismo peso en el cálculo de distancias de K-Means.
Determinación de k: Se utilizó el Método del Codo. Adjuntar (o describir) el gráfico del codo generado y justificar la elección de optimal_k. Mencionar que se eligió k= (el valor que hayas seleccionado, ej: 4) porque es el punto donde la disminución de la inercia se vuelve menos pronunciada.
Resultados del Clustering:
Mostrar la tabla de características promedio por clúster (cluster_summary).
Interpretación de cada clúster: Basado en los promedios y las visualizaciones, dar un nombre o descripción a cada grupo. Por ejemplo: "Clúster 0: Estaciones Locales de Baja Afluencia", "Clúster 1: Grandes Hubs de Transferencia", "Clúster 2: Estaciones Suburbanas/Residenciales", "Clúster 3: Estaciones de Afluencia Media". Esta es la parte más importante del análisis.
Incluir o describir las visualizaciones generadas (pairplot, scatter plots), explicando cómo ayudan a ver la separación de los grupos.
Limitaciones: Mencionar que se usaron datos sintéticos, por lo que los resultados son ilustrativos. Con datos reales, los patrones podrían ser diferentes. K-Means asume clústeres esféricos y de tamaño similar, lo cual podría no ser siempre el caso.
