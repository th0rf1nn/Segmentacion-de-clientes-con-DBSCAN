# Segmentación de clientes con DBSCAN

## 1. Técnica utilizada y justificación
Para esta actividad optamos por usar **DBSCAN** en lugar de métodos más tradicionales como K-Means.

**¿Por qué esta elección?**
Principalmente porque en riesgo crediticio los grupos de clientes no suelen tener formas geométricas perfectas. DBSCAN tiene una ventaja clave para nosotros: sabe diferenciar entre "grupos densos" y "ruido" (outliers). Esto nos permite aislar a los clientes con comportamientos atípicos en lugar de forzarlos a pertenecer a un segmento promedio, lo cual es vital para detectar posibles fraudes o nichos específicos.

**Datos usados:**
Trabajamos con el set de entrenamiento procesado. Le agregamos valor mediante Feature Engineering, cruzando la tabla principal con los históricos del buró (`bureau` y `bureau_balance`). Generamos variables agregadas (como promedios de deuda y máximos días de mora) para que el clustering tuviera más contexto financiero y no solo demográfico.

## 2. Instrucciones de ejecución
El desarrollo completo está en el notebook `ML_prueba_3.ipynb`.

**Requisitos:**
* Python 3.x
* Librerías estándar: pandas, numpy, seaborn, scikit-learn.
* Importante: Tener instalada la librería `pyarrow` para leer los archivos parquet.

**Pasos:**
1.  Asegúrate de que los archivos `application_.parquet`, `bureau.parquet` y `bureau_balance.parquet` estén en la misma carpeta que el notebook.
2.  Ejecuta las celdas en orden. El código se encarga de:
    * Cargar y limpiar los datos (imputación de nulos).
    * Estandarizar las variables con `StandardScaler` (paso crítico para DBSCAN).
    * Calcular la gráfica de distancias para definir el radio (Epsilon).
    * Entrenar el modelo y visualizar los resultados con PCA.

## 3. Análisis de los resultados
Para configurar el modelo usamos el gráfico de codo de los vecinos más cercanos, fijando un **Epsilon de 1.2** y un **MinPts de 30**.

**Lo que encontramos:**
El algoritmo detectó **5 clusters** definidos y separó un grupo relevante de ruido.

* **Cluster 0 (El perfil estándar):** Agrupa a la gran mayoría (212.112 clientes). Tienen una tasa de mora del **9.05%**, muy cercana al promedio general. Este es nuestro cliente tipo.
* **Cluster 3 (Alto Riesgo):** Es un grupo pequeño (43 casos) pero con la mora más alta (**11.6%**). Aquí hay una combinación de variables que claramente indica problemas de pago.
* **Los Outliers (Etiqueta -1):** Esto fue lo más interesante del análisis. El algoritmo marcó como "ruido" a cerca del **15%** de la base (46.842 clientes). Contra lo que uno podría pensar, este grupo tiene una mora del **6.4%**, bastante mejor que el promedio.

**Interpretación:**
Los datos nos dicen que ser un cliente "atípico" en este banco no significa ser riesgoso. Probablemente son personas con estructuras de ingresos o deudas poco comunes (quizás emprendedores o rentas altas complejas), pero que tienen buen comportamiento de pago.

## 4. Conclusión e incorporación al proyecto
**¿Se incorpora al modelo final? SÍ.**

La recomendación es incluir la etiqueta del cluster (`CLUSTER_LABEL_DBSCAN`) como una nueva variable en el modelo supervisado por dos razones:

1.  **Mejora la discriminación:** El modelo supervisado (como un XGBoost) podrá aprender que si un cliente cae en el "Cluster 3", debe penalizar su score, mientras que si es "Outlier" (-1), podría bonificarlo o al menos no castigarlo por tener datos extremos.
2.  **Limpieza de señal:** Al identificar explícitamente a los outliers que pagan bien, evitamos que el modelo se confunda y rechace a buenos clientes solo por tener variables fuera de rango. Esto le da más estabilidad a la predicción final.