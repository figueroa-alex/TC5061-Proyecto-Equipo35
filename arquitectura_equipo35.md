# Arquitectura del Sistema y Pipeline de ML — Proyecto MiBici (Equipo 35)

Documento de referencia del **Equipo 35** para la materia **TC5061 — Operaciones de Aprendizaje Automático**. Describe el diagrama de arquitectura de la solución, el flujo del pipeline de entrenamiento por pregunta y la justificación técnica de las herramientas seleccionadas.

---

## 1. Diagrama de Arquitectura del Sistema

```text
┌────────────────────────────────────────────────────────────────────────────────────┐
│                   CAPA 1: FUENTES DE DATOS Y ALMACENAMIENTO LOCAL                  │
│                                                                                    │
│  [Portal Web MiBici] ──(urllib)──► data/raw/datos_mibici/AAAA/*.csv (48 meses)     │
│                                    data/raw/nomenclatura/2026/nomenclatura.csv     │
│                                                   │                                │
│                                    (mibici/dataset.py + PyArrow)                   │
│                                                   ▼                                │
│                            data/processed/datos_mibici.parquet (17.03M filas)      │
│                            data/processed/data_version.json (SHA-256: 128979137b46)│
└───────────────────────────────────────────────────┬────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                CAPA 2: PROCESAMIENTO, FEATURE ENGINEERING Y MODELADO              │
│                                                                                   │
│  ┌──────────────────────┐   ┌───────────────────────┐   ┌──────────────────────┐  │
│  │   mibici/config.py   │──►│  mibici/features.py   │──►│mibici/modeling/train │  │
│  │ • Rutas Pathlib      │   │ • Tabla EstaciónxHora │   │ • Partición temporal │  │
│  │ • Fechas 2022-2026   │   │ • Rezagos sin leakage │   │ • P1: Ridge / Dummy  │  │
│  │ • Semilla fija (42)  │   │ • Perfil estaciones P3│   │ • P2: LogReg / Dummy │  │
│  └──────────────────────┘   └───────────────────────┘   │ • P3: KMeans K=4/K=2 │  │
│                                                         └──────────┬───────────┘  │
└────────────────────────────────────────────────────────────────────┼──────────────┘
                                                                     │
          ┌──────────────────────────────────────────────────────────┴─────┐
          ▼                                                                ▼
┌─────────────────────────────────────────┐    ┌────────────────────────────────────┐
│   CAPA 3: EXPERIMENT TRACKING (MLFlow)  │    │  CAPA 4: COLABORACIÓN Y GOBERNANZA │
│                                         │    │                                    │
│  • Backend: sqlite:///mlflow.db         │    │  • Cookiecutter Data Science (ccds)│
│  • Experimento: MiBici_Base_Equipo35    │    │  • Entorno aislado (.venv +        │
│  • Parámetros: hiperparámetros, semilla,│    │    requirements.txt con ==)        │
│    rango de fechas y data_version       │    │  • Git / GitHub Flow:              │
│  • Métricas con prefijo: train_, val_,  │    │    Rama 'main' protegida +         │
│    test_, baseline_                     │    │    4 Feature Branches +            │
│  • Tags: equipo, pregunta, git_commit   │    │    Pull Requests con revisión      │
│  • Artefactos: figuras PNG en reports/  │    │    cruzada entre los 4 integrantes │
└─────────────────────────────────────────┘    └────────────────────────────────────┘