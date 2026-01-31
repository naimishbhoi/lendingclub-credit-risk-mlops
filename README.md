# End-to-End Credit Risk Modeling System Using LendingClub Data

This is an end-to-end machine learning project for credit risk modelling using historical consumer loan data from LendingClub (2007–2018).

---

## System Architecture Flow

```text
Raw Data
   │
   ▼
Ingestion ──► Validation ──► Feature Engineering
   │                              │
   │                              ▼
   │                         Feature Artifacts
   │                              │
   ▼                              ▼
Training ──► Tuning ──► Evaluation ──► Promotion Policy
                                            │
                                            ▼
                                      Model Registry
                                            │
                   ┌────────────────────────┴────────────────────────┐
                   ▼                                                 ▼
             Batch Inference                                   Online Inference
                   │                                                 │
                   ▼                                                 ▼
             Predictions                                      Web/API Consumers
                   │                                                 │
                   └──────────────► Monitoring ◄────────────────────┘
                                      │
                                      ▼
                                 Retraining
```
---

## Repository Structure

```text
lendingclub-credit-risk-mlops/
├── configs/
│   ├── data.yaml              # Data source paths, splits, time windows
│   ├── features.yaml          # Feature-related configuration knobs
│   ├── training.yaml          # Model training parameters
│   ├── tuning.yaml            # Hyperparameter search configuration
│   ├── evaluation.yaml        # Metrics, thresholds, promotion rules
│   ├── inference.yaml         # Batch / online inference settings
│   ├── monitoring.yaml        # Drift, performance, alert thresholds
│   └── paths.yaml             # Centralized artifact & data paths
│
├── data/
│   ├── raw/                   # Immutable raw datasets (source of truth)
│   ├── interim/               # Deterministic intermediate transforms
│   ├── processed/             # Model-ready datasets
│   └── external/              # Third-party reference data (if any)
│
├── artifacts/
│   ├── features/              # Serialized feature matrices
│   ├── models/                # Trained model artifacts
│   ├── metrics/               # Evaluation results per run
│   ├── predictions/           # Batch inference outputs
│   └── metadata/              # Run metadata (hashes, configs, lineage)
│
├── src/
│   ├── contracts/
│   │   ├── dataset_contract.yaml   # Dataset acceptance & constraints
│   │   ├── feature_contract.yaml   # Authoritative feature definitions (v1)
│   │   └── target_contract.yaml    # Target & label derivation rules
│   │
│   ├── schemas/
│   │   ├── raw_data.py         # Raw dataset schema validation
│   │   ├── feature_schema.py   # Feature matrix schema
│   │   └── prediction_schema.py# Inference input/output schemas
│   │
│   ├── ingestion/
│   │   ├── ingest.py           # Raw data loading logic
│   │   └── validate.py         # Dataset validation against contracts
│   │
│   ├── features/
│   │   ├── transformers/
│   │   │   ├── base.py         # Base transformer interface
│   │   │   ├── numerical.py    # Numerical feature transformations
│   │   │   ├── categorical.py  # Categorical feature transformations
│   │   │   └── temporal.py     # Time-derived feature logic
│   │   │
│   │   ├── build_features.py   # Orchestrates feature generation
│   │   └── schema.py           # Maps contracts to feature schemas
│   │
│   ├── training/
│   │   ├── estimator.py        # Model factory (algorithm selection)
│   │   ├── trainer.py          # Model fitting & callbacks
│   │   ├── tuning.py           # Hyperparameter optimization logic
│   │   └── train.py            # Training CLI entrypoint
│   │
│   ├── evaluation/
│   │   ├── evaluate.py         # Metric computation
│   │   ├── checks.py           # Statistical & regression checks
│   │   └── promote.py          # Policy-based model promotion logic
│   │
│   ├── registry/
│   │   ├── base.py             # Abstract model registry interface
│   │   ├── local.py            # Local filesystem registry
│   │   ├── mlflow.py           # MLflow-backed registry
│   │   └── register.py         # Registry orchestration logic
│   │
│   ├── inference/
│   │   ├── batch.py            # Offline / batch prediction logic
│   │   └── service.py          # Online inference service adapter
│   │
│   ├── monitoring/
│   │   ├── checks/
│   │   │   ├── data_quality.py # Data quality monitoring
│   │   │   ├── drift.py        # Feature & prediction drift detection
│   │   │   └── performance.py  # Model performance tracking
│   │   │
│   │   ├── alerts/
│   │   │   ├── slack.py        # Slack alert integration
│   │   │   └── email.py        # Email alert integration
│   │   │
│   │   └── run_monitoring.py   # Monitoring pipeline entrypoint
│   │
│   ├── common/
│   │   ├── config.py           # Central config loader & validator
│   │   ├── logging.py          # Standardized logging setup
│   │   ├── exceptions.py       # Custom exception hierarchy
│   │   ├── io.py               # File & artifact I/O utilities
│   │   └── utils.py            # Shared helper functions
│
├── pipelines/
│   ├── validate_pipeline.py    # Dataset & schema validation pipeline
│   ├── train_pipeline.py       # End-to-end training pipeline
│   ├── inference_pipeline.py   # End-to-end inference pipeline
│   └── retrain_pipeline.py     # Monitoring-triggered retraining
│
├── webapp/
│   ├── api/
│   │   ├── routes.py           # API route definitions
│   │   ├── schemas.py          # Request/response schemas
│   │   └── predictor.py        # Thin inference wrapper (no ML logic)
│   │
│   ├── ui/
│   │   ├── templates/
│   │   │   └── index.html      # Simple UI for predictions
│   │   └── static/
│   │       └── style.css       # UI styling
│   │
│   └── run.py                  # Web application entrypoint
│
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Pipeline & system tests
│   └── data/                   # Test fixtures
│
├── scripts/
│   ├── run_training.sh         # Local training execution
│   ├── run_inference.sh        # Local inference execution
│   └── run_monitoring.sh       # Local monitoring execution
│
├── docker/
│   ├── training.Dockerfile     # Training environment image
│   └── serving.Dockerfile      # Inference service image
│
├── infra/                      # Infrastructure-specific configs
│   ├── mlflow/                 # MLflow setup (tracking/registry)
│   ├── logging/                # Logging infra configs
│   └── docker/                 # Shared docker assets
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Linting, tests, sanity checks
│       └── cd.yml              # Deployment & promotion automation
│
├── pyproject.toml              # Dependencies & tooling config
├── Makefile                    # Common developer commands
└── README.md                   # Project documentation (written later)

```
---