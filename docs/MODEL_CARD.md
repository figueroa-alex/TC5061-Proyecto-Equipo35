# Model Card — Sistema Predictivo MiBici (Equipo 35)

Documentación responsable y de gobernanza basada en el estándar *Model Cards for Model Reporting* (Mitchell et al., 2019) para los tres modelos de Machine Learning desplegados en producción en el proyecto integrador **TC5061-Proyecto-Equipo35**.

---

## 1. Detalles de los Modelos (*Model Details*)

* **Organización / Desarrolladores:** Equipo 35 — Maestría en Inteligencia Artificial Aplicada (TC5061.10 Operaciones de Aprendizaje Automático, Tecnológico de Monterrey):
  * Alejandro Figueroa Martínez (A01796930) — Científico de Datos
  * Fernando Gómez Moreno (A00354097) — Ingeniero de Datos
  * Luis Pablo Guerrero Garza (A01840428) — Ingeniero de Machine Learning
  * Rosina Flores Ruiz (A01639950) — Analista de Datos
* **Fecha de documentación:** Noviembre de 2026.
* **Repositorio oficial:** `https://github.com/figueroa-alex/TC5061-Proyecto-Equipo35`
* **Licencia:** Definida en el archivo `LICENSE` en la raíz del repositorio.
* **Estrategia de versionado y ciclo de vida:** MLflow Model Registry con esquema vigente de **Aliases** (`@champion` en producción y `@challenger` como retador / respaldo de reversión), sirviéndose mediante carga híbrida en FastAPI (`models:/<nombre>@champion` con respaldo empaquetado en `models/registry_bundle/`).

### Inventario y linaje de modelos en producción (`@champion`)

| Código | Nombre en Model Registry | Tarea y Algoritmo | Versión `@champion` (y `@challenger`) | Ventana y Versión de Datos (`SHA-256`) | Run ID de Origen (`mlflow.db`) |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **P1** | `mibici-equipo35-p1-demanda` | **Regresión** (`HistGradientBoostingRegressor`: `max_iter=300`, `max_depth=12`, `lr=0.06`, `min_samples_leaf=30`, `sample_weight=3.0` en 2026) | **`v3`** (`v2` en `@challenger`) | Oct 2022 – Sep 2026 (48 meses) + `nomenclatura_2026_10.csv` (`128979137b46+3542d1d567e1`) | `ffbea1f30aaf41d98c52f371a6fea813` |
| **P2** | `mibici-equipo35-p2-riesgo` | **Clasificación binaria** (`HistGradientBoostingClassifier`: `max_iter=250`, `max_depth=12`, `lr=0.08`, `class_weight="balanced"`) | **`v2`** (`v3` en `@challenger`) | Oct 2022 – Dic 2025 (39 meses) + `nomenclatura_2026_09.csv` (`128979137b46`) | `3bcd2d0dc86c42feae230ac3cf8d9e1c` |
| **P3** | `mibici-equipo35-p3-grupos` | **Clustering** (`Pipeline`: `StandardScaler` + `KMeans(n_clusters=4, n_init=20)`) | **`v3`** (`v2` en `@challenger`) | Perfiles Ene 2026 – Sep 2026 (469 estaciones activas) + `nomenclatura_2026_10.csv` (`128979137b46+3542d1d567e1`) | `dcbfa909e35f4616befcf6c1fed3fd12` |

*(Nota sobre linaje de versiones iniciales: la versión `v1` de cada modelo corresponde a los algoritmos retadores iniciales `RandomForestRegressor` en P1, `RandomForestClassifier` en P2 y `MiniBatchKMeans` en P3).*

---

## 2. Uso Previsto (*Intended Use*)

### Casos de uso primarios y usuarios finales
* **P1 — Planeación operativa de flota (Operador central de MiBici):** Predecir el número de viajes que iniciarán en cada estación durante la siguiente hora ($t$) para dimensionar la flota activa y programar ventanas de mantenimiento preventivo en horas valle.
* **P2 — Priorización de rebalanceo (Equipo logístico de camionetas):** Alertar con 1 hora de anticipación si una estación presenta riesgo de desbalance operativo ($\ge 30\%$ de su capacidad instalada `dpcapacity`) para despachar vehículos de redistribución antes de que se vacíe o se sature.
* **P3 — Planeación territorial y expansión (Dirección de Movilidad y Planeación Urbana):** Clasificar las estaciones del Área Metropolitana de Guadalajara (AMG) en 4 perfiles funcionales para diseñar políticas diferenciadas de capacidad de anclajes y nuevas instalaciones.

### Usos fuera de alcance (*Out-of-Scope Uses*)
* **Perfilamiento individual de ciclistas:** Los modelos **no** deben emplearse para inferir comportamientos, rutas, edad o género de personas usuarias. El contrato de entrada en FastAPI (`extra="forbid"`) bloquea cualquier intento de enviar columnas personales.
* **Control automatizado sin supervisión humana:** Las alertas de P2 son un insumo de apoyo a la decisión para el centro de control; no deben disparar penalizaciones ni cierres automáticos de estaciones sin validación operativa.
* **Transferencia directa a otras ciudades o periodos atípicos sin reentrenamiento:** Los modelos dependen de las coordenadas geográficas, capacidades y hábitos post-pandemia del AMG (Guadalajara, Zapopan y Tlaquepaque); no son válidos fuera de ese dominio ni ante disrupciones estructurales (cierres masivos o contingencias sanitarias).

---

## 3. Factores Relevantes (*Factors*)

Los factores que más influyen en el comportamiento y desempeño de los modelos son:
* **Estacionalidad temporal y climática:** Hora del día (horas pico `07:00–09:59` y `17:00–19:59`), día de la semana (días laborales vs. fin de semana) y transiciones climáticas del AMG (temporada de lluvias junio–septiembre vs. octubre seco con mayor actividad recreativa y Fiestas de Octubre).
* **Infraestructura física de la estación:** Capacidad instalada (`dpcapacity`, mediana de `15` anclajes y media de `17.14` en octubre de 2026) y ubicación geográfica (`latitude`, `longitude`). Cambios de catálogo —como el retiro de `14` estaciones y el ajuste de anclajes en `11` estaciones entre septiembre y octubre de 2026— modifican directamente la tasa de desbalance.
* **Dinámica de red inmediata ($t-1$):** En P1 y P2, los rezagos de salidas, llegadas, flujo neto y promedio móvil de 3 horas (`lag_1_iniciados`, `lag_2_iniciados`, `lag_24_iniciados`, `media_movil_3_iniciados`, `lag_1_terminados`, `lag_1_flujo_neto` y `lag_1_ratio_capacidad`) son los principales impulsores de la predicción.
* **Vocación territorial del polígono (P3):** Diferencias entre el núcleo central de alta rotación (Centro Histórico–Americana–Chapultepec), corredores emisores matutinos (Oriente–Tlaquepaque), corredores vespertinos (Poniente–Chapalita–Providencia) y el polígono Norte (Zapopan Centro).

---

## 4. Métricas (*Metrics*)

| Modelo | Métrica Primaria | Métricas Secundarias | Criterio de Decisión Operativa |
| :--- | :--- | :--- | :--- |
| **P1 (Regresión)** | **MAE** (error absoluto medio en viajes/hora) | `MAE en Hora Pico`, `RMSE`, $R^2$, `Sesgo Medio (Bias)` | Reducir el error directamente en bicicletas/hora, minimizando la subestimación en picos. |
| **P2 (Clasificación)** | **F1-Score** (umbral `0.50`) y **PR-AUC** | `Recall`, `Precision`, `ROC-AUC`, `Brier Score` | Maximizar la detección oportuna de desbalances raros (`~3%–4.6%` de prevalencia) sin disparar falsas alarmas no calibradas. No se usa *Accuracy* por desbalance severo de clases. |
| **P3 (Clustering)** | **Distancia media al centroide** y **Silhouette Score** | `Davies-Bouldin Index`, `Proporción del clúster mínimo` | Lograr grupos compactos, interpretables y balanceados (evitando el colapso en un solo grupo). |

**Protocolo estadístico de comparación entre versiones (A/B Testing):** Dado que las observaciones estación $\times$ hora están correlacionadas en el tiempo (rezagos horarios) y en el espacio (efectos de red por día y por estación), la incertidumbre entre `v2` y `v3` se estima mediante **remuestreo *bootstrap* por bloques** (`2,000` réplicas en métricas primarias, `300` en PR-AUC y `500` en Silhouette; semilla fija `42`), remuestreando sobre los **31 días** y las **469 estaciones** de octubre de 2026 con intervalos de confianza al 95% y $p$-valor bilateral con corrección de continuidad.

---

## 5. Datos de Entrenamiento y Evaluación (*Training & Evaluation Data*)

* **Fuente:** Portal oficial de Datos Abiertos de MiBici (`https://mibici.net/es/datos-abiertos/`), publicados por la Agencia Metropolitana de Servicios de Infraestructura para la Movilidad (AMIM) bajo libre uso con atribución de fuente.
* **Bloque histórico base (`2022-10` a `2026-09`, 48 meses — `data_version: 128979137b46`):**
  * **Justificación del inicio:** Octubre de 2022 marca la abrogación de medidas sanitarias obligatorias por COVID-19 en Jalisco (`DIELAG ACU 065/2022`), evitando *concept drift* de confinamiento.
  * **Volumen y limpieza:** `17,927,033` registros crudos $\to$ **`17,036,839` viajes limpios** en `datos_mibici.parquet` (eliminación de `889,481` viajes $< 121$ s y `713` duplicados exactos; homologación de codificación `utf-8-sig`/`cp1252` y formatos de fecha; estandarización de `Genero` y `Año_de_nacimiento`).
  * **Partición cronológica estricta (sin fuga de futuro, `6,628,480` filas estación $\times$ hora):**
    * *Train:* Oct 2022 – Dic 2025 (39 meses).
    * *Validation:* Ene 2026 – May 2026 (5 meses).
    * *Test:* Jun 2026 – Sep 2026 (4 meses, `691,406` observaciones).
* **Lote independiente de producción (`2026-10` — `data_version: 3542d1d567e1`):**
  * `635,532` viajes crudos $\to$ **`589,672` viajes limpios** cruzados con `nomenclatura_2026_10.csv` (`470` estaciones en catálogo, `469` activas), conformando **`208,912` observaciones estación $\times$ hora** (`31` días) para monitoreo con Evidently y A/B Testing.
* **Validación automatizada de datos:** `DeepChecks` (integridad, `0` duplicados, `0.0%` de fuga entre Train y Test en P1, P2 y P3) y `Evidently v0.7.23` (`0` valores nulos y `0` filas duplicadas en referencia y producción).

---

## 6. Análisis Cuantitativo (*Quantitative Analyses*)

### 6.1 Desempeño frente a Baselines en el periodo de desarrollo (Test: Jun–Sep 2026)

| Modelo | Baseline (Test) | Modelo Candidato Inicial `v2` (Test) | Mejora sobre Baseline |
| :--- | :--- | :--- | :--- |
| **P1 (Demanda)** | `DummyRegressor`: `MAE = 1.8358`, `RMSE = 2.8032`, $R^2 = -0.0059$ | `HistGB v2`: **`MAE = 1.1761`**, `RMSE = 1.7609`, $R^2 = 0.6031$ | **`+35.94%`** reducción en MAE |
| **P2 (Riesgo)** | `DummyClassifier`: `F1 = 0.0360`, `Recall = 0.0461`, `ROC-AUC = 0.5000`, `PR-AUC = 0.0295` | `HistGB_Clf v2`: **`F1 = 0.1727`**, `Recall = 0.7987`, `Precision = 0.0968`, `ROC-AUC = 0.8677`, `PR-AUC = 0.2240` | **`4.80×`** en F1 y **`7.59×`** en PR-AUC |
| **P3 (Grupos)** | `KMeans K=2` (Train 373 est.): `Silhouette = 0.2288`, `Inercia = 2614.77` | `KMeans K=4 v2` (Train 373 est.): `Silhouette = 0.2028`, `Davies-Bouldin = 1.6270`, `Inercia = 2005.95`, `Min share = 9.92%` | **`-23.28%`** dispersión intra-grupo con 4 perfiles urbanos claros |

### 6.2 Evaluación en producción y A/B Testing con Bootstrap por Bloques (Octubre 2026)

Sobre las **`208,912` observaciones horarias (`31` días, `469` estaciones)** de octubre de 2026:

| Modelo | Métrica | Variante A (`v2`) | Variante B (`v3`) | Diferencia (`v3 - v2`) | Bootstrap por Días (31) — IC 95% / $p$ | Bootstrap por Estaciones (469) — IC 95% / $p$ | Decisión (`@champion`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P1** | `MAE` Global<br>`MAE` Pico / `RMSE` / $R^2$ / `Bias` | `1.3572`<br>`1.5718` / `2.1109` / `0.6511` / `-0.3003` | **`1.3427`**<br>**`1.5537`** / **`2.0666`** / **`0.6656`** / **`-0.2810`** | **`-0.0144`** | `[-0.0175, -0.0116]`<br>$p = 0.0010$ *(Sig.)* | `[-0.0212, -0.0089]`<br>$p = 0.0010$ *(Sig.)* | **Promueve `v3`** (`v2` $\to$ `@challenger`) |
| **P2** | `F1-Score` (umbral `0.50`)<br>`Brier Score` (calibración)<br>`PR-AUC` (`Recall`: `0.8055` vs `0.8280`) | **`0.1913`**<br>**`0.1563`**<br>`0.2448` | `0.1873`<br>`0.1649`<br>**`0.2512`** | `-0.0040`<br>`+0.0086`<br>`+0.0064` | `[-0.0049, -0.0031]`, $p = 0.0010$<br>`[+0.0083, +0.0088]`, $p = 0.0010$<br>`[+0.0048, +0.0082]`, $p = 0.0066$ | `[-0.0058, -0.0023]`, $p = 0.0010$<br>`[+0.0072, +0.0101]`, $p = 0.0010$<br>`[+0.0019, +0.0103]`, $p = 0.0199$ | **Protege `v2`** (`v3` $\to$ `@challenger` para calibrar umbral) |
| **P3** | `Dist. Centroide`<br>`Silhouette` (`DB`: `1.8117` vs `1.7937`)<br>`Min. Cluster Share` | `2.3685`<br>`0.1533`<br>`10.45%` | **`2.3142`**<br>`0.1601`<br>**`19.19%`** | **`-0.0543`**<br>`+0.0068` | N/A (perfil por estación) | `[-0.1009, -0.0097]`, $p = 0.0180$ *(Sig.)*<br>`[-0.0028, +0.0149]`, $p = 0.1996$ *(NS)* | **Promueve `v3`** (menor distancia y fin del colapso de grupos) |

### 6.3 Desempeño operativo bajo carga y eficiencia FinOps
* **Pruebas de carga con Locust (`0.00%` de errores en `5,957` peticiones totales):**
  * *Carga normal (10 usuarios, 30 s):* `19.89 RPS` totales (`16.11 RPS` ML), latencia P50 `5.0 ms`, P95 `12.0 ms`, P99 `27.0 ms`.
  * *Carga pico (50 usuarios, 60 s):* `90.98 RPS` totales (`73.45 RPS` ML), latencia P50 `35.0 ms`, P95 `130.0 ms` (bajo el SLO de `200 ms`), P99 `280.0 ms`.
* **Consumo medido (`podman stats` en pico, 4 microservicios):** `953.76 MB` de RAM pico y `37.74%` de CPU pico combinada (tras reducir `mibici-mlflow` de `2,041 MB` a `428.3 MB` desactivando el ejecutor interno de trabajos).
* **Economía FinOps (AWS `us-east-1` + 20 GB EBS `gp3`):** Reducción de `$31.97 USD/mes` (`t3.medium` OD, `$60.82 / 1M` inferencias en demanda base de `525,600` inf./mes) a **`$9.63 USD/mes`** (`t4g.small` ARM64 con Reserved Instance a 1 año, ahorro del **`69.9%`**, **`$18.32 / 1M`** inferencias).

---

## 7. Consideraciones Éticas, Privacidad y Equidad (*Ethical Considerations*)

1. **Privacidad desde el diseño (*Privacy by Design*):** Los datos abiertos de MiBici no contienen PII directo, pero incluyen un identificador seudonimizado (`Usuario_Id`), `Genero` y `Año_de_nacimiento`. Por política de gobernanza del equipo, estas tres columnas **nunca** se utilizan como variables predictoras, **nunca** se registran en los conjuntos de entrada de MLflow (`mlflow.log_input`) y son rechazadas explícitamente con código HTTP `422` por Pydantic (`extra="forbid"`) en la API REST.
2. **Equidad territorial (*Geographic Fairness*):** Las estaciones del polígono central (Centro Histórico, Americana y Chapultepec) concentran mayor volumen histórico que las zonas periféricas o las estaciones recién instaladas en 2026. Si el equipo operativo asignara camionetas únicamente por volumen absoluto (P1), podría subatender colonias periféricas con menor densidad de viajes. Para mitigar este sesgo, el sistema combina **P1** con **P2** (que mide el desbalance de forma **relativa a la capacidad de cada estación**) y **P3** (segmentación por polígono urbano).
3. **Seguridad de software y cadena de suministro:** Suite automatizada de **47 pruebas** (`pytest` en modo real y sintético), análisis estático con `bandit -r mibici/` (`0` alertas), auditoría de dependencias con `pip-audit` (`0` vulnerabilidades conocidas), validación estricta de esquemas URL en la ingesta y ejecución del contenedor con usuario no privilegiado (`non-root`).

---

## 8. Limitaciones, Riesgos y Recomendaciones (*Caveats and Recommendations*)

### Limitaciones conocidas
* **Ausencia de inventario en tiempo real:** Los datos abiertos no reportan cuántas bicicletas hay ancladas minuto a minuto; por ello P2 aproxima el riesgo mediante el desbalance neto horario $\ge 30\%$ de `dpcapacity`.
* **Variables exógenas no observadas:** Los modelos no consumen pronósticos meteorológicos en tiempo real (lluvias súbitas), cierres viales ni calendario dinámico de eventos masivos (conciertos o marchas), lo cual explica parte de la subestimación residual en picos extraordinarios de P1 (`bias = -0.2810` viajes/h en octubre de 2026).
* **Calibración del umbral en P2:** Debido a que solo `3.66%` de las horas-estación de octubre de 2026 fueron de riesgo real y se usó `class_weight="balanced"`, la precisión puntual al umbral fijo `0.50` es baja (`10.86%` en `v2` y `10.56%` en `v3`). Aunque el ordenamiento probabilístico es alto (`ROC-AUC = 0.8574` en `v2` y `0.8622` en `v3`), se recomienda consumir P2 priorizando el *ranking* de probabilidad o ajustando un umbral operativo $\tau > 0.50$ acorde al número de camionetas disponibles por turno.
* **Incertidumbre en métricas de clustering (P3):** El *bootstrap* por estaciones mostró que la mejora de Silhouette Score de `v3` (`+0.0068`, `p = 0.1996`) no es estadísticamente significativa por sí sola; las decisiones de planeación deben complementarse con inspección geográfica y conocimiento territorial.

### Gobernanza y plan de mantenimiento (*Continuous Training*)
* **Monitoreo mensual con Evidently:** Al publicarse cada corte mensual de MiBici y su catálogo de nomenclatura, ejecutar `make drift-report`. En octubre de 2026 se detectó *drift* en `9/19` columnas de P1/P2 (`47.37%`, demanda `+17.50%`) y `6/10` columnas de P3 (`60.00%`, salidas diarias `+29.67%`).
* **Disparadores de reentrenamiento:** Activar `make ab-test` si `drift_share >= 0.35`, si la demanda media se desvía más de $\pm 10\%$ respecto a la predicción o si cambia la nomenclatura de estaciones.
* **Compuerta obligatoria de promoción:** Ninguna versión nueva reemplaza al `@champion` automáticamente sin superar el A/B Testing con **bootstrap por bloques (días y estaciones)** en la métrica operativa primaria. Si ocurre degradación en producción, se ejecuta *rollback* inmediato reasignando el alias `@champion` hacia `@challenger` en el Model Registry.

---

## 9. Referencias

* Huyen, C. (2022). *Designing machine learning systems: An iterative process for production-ready applications*. O'Reilly Media.
* MiBici. (2026). *Datos abiertos del sistema de bicicletas públicas MiBici* [Conjunto de datos]. Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara. https://mibici.net/es/datos-abiertos/
* Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19)* (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596
* Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the enterprise*. O'Reilly Media.