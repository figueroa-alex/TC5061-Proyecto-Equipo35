# Machine Learning Canvas y Formulación de Preguntas — Proyecto MiBici (Equipo 35)

Documento de referencia del **Equipo 35** para la materia **TC5061 — Operaciones de Aprendizaje Automático**. Contiene el diseño del **Machine Learning Canvas** y la especificación formal de las tres preguntas analíticas del sistema (**P1, P2 y P3**).

---

## 1. Machine Learning Canvas del Proyecto MiBici

| Bloque del ML Canvas | Definición Operativa y Técnica (Equipo 35) |
| :--- | :--- |
| **1. Propuesta de Valor (Value Proposition)** | Optimizar la operación diaria y la planeación territorial del sistema de bicicleta pública **MiBici** en el Área Metropolitana de Guadalajara (Guadalajara, Zapopan y Tlaquepaque) mediante tres capacidades de aprendizaje automático: (1) pronosticar la demanda horaria de viajes iniciados por estación (**P1**), (2) alertar anticipadamente el riesgo de vaciado o saturación de estaciones para guiar el rebalanceo logístico (**P2**), y (3) agrupar las estaciones según sus patrones de uso y ubicación geográfica para orientar políticas de expansión de infraestructura (**P3**). |
| **2. Usuarios Finales y Decisiones (Decisions & Users)** | • **P1 (Operador central del sistema):** Dimensionar la flota activa por turno horario y programar mantenimientos preventivos en horas valle.<br>• **P2 (Equipo logístico de rebalanceo):** Priorizar el envío de camionetas abastecedoras hacia estaciones en riesgo antes de que se queden sin bicicletas o sin puertos de anclaje.<br>• **P3 (Dirección de Movilidad y Planeación Urbana):** Definir en qué zonas instalar nuevas estaciones o ampliar la capacidad de anclajes (`dpcapacity`) según el perfil funcional de cada grupo urbano. |
| **3. Fuentes de Datos (Data Sources)** | • **Histórico de viajes abiertos (`datos_abiertos_AAAA_MM.csv`):** 48 archivos mensuales desde octubre de 2022 hasta septiembre de 2026 (`17,927,033` registros crudos; `17,036,839` registros limpios consolidados en `data/processed/datos_mibici.parquet`, versión SHA-256 `128979137b46`).<br>• **Catálogo maestro de nomenclatura (`nomenclatura_2026_09.csv`):** `484` estaciones registradas con `id`, `name`, `latitude`, `longitude` y `dpcapacity` (`473` estaciones con viajes activos en el periodo). |
| **4. Variables de Entrada (Features)** | • **Para P1 y P2 (Granularidad Estación × Hora):** `hora` (0–23), `dia_semana` (0–6), `mes` (1–12), `es_fin_de_semana`, `es_hora_pico` (7–9 h y 17–19 h), `dpcapacity`, `latitude`, `longitude`, rezagos estrictamente pasados de salidas (`lag_1_iniciados`, `lag_2_iniciados`, `lag_24_iniciados`), promedio móvil de las 3 horas previas (`media_movil_3_iniciados`), llegadas en la hora previa (`lag_1_terminados`), flujo neto previo (`lag_1_flujo_neto`) y razón de desbalance previo respecto a capacidad (`lag_1_ratio_capacidad`).<br>• **Para P3 (Granularidad Estación):** `promedio_salidas_diarias`, `duracion_mediana_min`, `prop_fin_de_semana`, `prop_pico_manana`, `prop_pico_tarde`, `ratio_salidas_llegadas`, `dpcapacity`, `latitude` y `longitude`. |
| **5. Tareas de Aprendizaje Automático (ML Tasks)** | • **P1 (Regresión supervisada):** Predecir el número de viajes iniciados `viajes_iniciados` en la estación $i$ durante la hora $t$.<br>• **P2 (Clasificación binaria supervisada):** Predecir la etiqueta `en_riesgo` $\in \{0, 1\}$ en la estación $i$ durante la hora $t$, donde `en_riesgo = 1` cuando $\Vert{}\text{viajes\_iniciados}_{i,t} - \text{viajes\_terminados}_{i,t}\Vert{} / \text{dpcapacity}_i \ge 0.30$.<br>• **P3 (Clustering no supervisado):** Segmentar las estaciones activas en $K=4$ grupos funcionales homogéneos. |
| **6. Evaluación Offline y Baselines (Offline Evaluation)** | • **P1:** Baseline `DummyRegressor(strategy="mean")` vs. `Ridge(alpha=1.0)`. Métricas: `MAE` (`test_mae = 1.3659` vs. baseline `1.8358`), `RMSE` (`test_rmse = 2.0732` vs. baseline `2.8032`) y `R2` (`test_r2 = 0.4498`).<br>• **P2:** Baseline `DummyClassifier(strategy="stratified")` vs. `LogisticRegression(class_weight="balanced")`. Métricas: `Recall` (`test_recall = 0.6753`), `ROC-AUC` (`test_roc_auc = 0.7970`), `F1-Score` (`test_f1 = 0.1487` vs. baseline `0.0360`) y `Precision` (`test_precision = 0.0835`).<br>• **P3:** Baseline `KMeans(n_clusters=2)` vs. `KMeans(n_clusters=4)`. Métricas: `Silhouette Score` (`0.2028`), `Davies-Bouldin` (`1.6270`) y coherencia espacial en mapa. |
| **7. Partición Temporal y Prevención de Fuga de Datos** | Partición estrictamente cronológica sin mezclar datos futuros:<br>• **Entrenamiento (Train):** `2022-10-01` a `2025-12-31` (39 meses).<br>• **Validación (Val):** `2026-01-01` a `2026-05-31` (5 meses).<br>• **Prueba (Test):** `2026-06-01` a `2026-09-30` (4 meses).<br>Todos los rezagos usan `.shift(1)` o superior y los escaladores (`StandardScaler`) se ajustan únicamente sobre `Train`. |
| **8. Inferencia, Monitoreo y Reentrenamiento** | Inferencia horaria por lotes para **P1** y **P2** con horizonte de 1 hora hacia adelante. Actualización mensual del archivo Parquet y reentrenamiento automático al publicarse el nuevo corte mensual de datos abiertos en el portal de MiBici. |

---

## 2. Resumen de Formulación por Pregunta

1. **P1 — Regresión (`P1_Regresion`):**
   * **Objetivo:** Estimar cuántos viajes se iniciarán por estación en la siguiente hora.
   * **Granularidad:** Estación $\times$ Hora.
   * **Impacto en la decisión:** Permite anticipar la demanda horaria con un error medio de `1.37` bicicletas por hora en prueba.

2. **P2 — Clasificación (`P2_Clasificacion`):**
   * **Objetivo:** Detectar si una estación experimentará en la siguiente hora un desbalance neto absoluto ($\Vert{}\text{salidas} - \text{llegadas}\Vert{}$) igual o superior al **30% de su capacidad instalada (`dpcapacity`)**.
   * **Granularidad:** Estación $\times$ Hora.
   * **Impacto en la decisión:** Identifica oportunamente el `67.53%` de los eventos críticos de vaciado o saturación (`test_roc_auc = 0.7970`), permitiendo priorizar las rutas de rebalanceo.

3. **P3 — Clustering (`P3_Clustering`):**
   * **Objetivo:** Agrupar las `373` estaciones activas del periodo de entrenamiento según su intensidad de uso, horarios pico, vocación emisora/receptora, capacidad y coordenadas.
   * **Granularidad:** Estación.
   * **Impacto en la decisión:** Separa claramente 4 zonas operativas (Cluster 0: Zapopan Norte con `37` estaciones; Cluster 1: Corredor Poniente con `93`; Cluster 2: Periferia Centro-Oriente con `133`; Cluster 3: Núcleo Central de alta rotación con `110`).