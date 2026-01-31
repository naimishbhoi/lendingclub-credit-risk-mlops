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
|
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── training.yaml
│   ├── tuning.yaml
│   ├── evaluation.yaml
│   ├── inference.yaml
│   ├── monitoring.yaml
│   └── paths.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── artifacts/
│   ├── features/
│   ├── models/
│   ├── metrics/
│   ├── predictions/
│   └── metadata/
│
├── src/
│   ├── ingestion/
│   │   ├── ingest.py
│   │   └── validate.py
│   │
│   ├── features/
│   │   ├── definitions/
│   │   │   └── feature_contract.yaml
│   │   │
│   │   ├── transformers/
│   │   │   ├── base.py
│   │   │   ├── numerical.py
│   │   │   ├── categorical.py
│   │   │   └── temporal.py
│   │   │
│   │   ├── build_features.py
│   │   └── schema.py
│   │
│   ├── training/
│   │   ├── estimator.py
│   │   ├── trainer.py
│   │   ├── tuning.py
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── checks.py
│   │   └── promote.py
│   │
│   ├── registry/
│   │   ├── base.py
│   │   ├── local.py
│   │   ├── mlflow.py
│   │   └── register.py
│   │
│   ├── inference/
│   │   ├── batch.py
│   │   └── service.py
│   │
│   ├── monitoring/
│   │   ├── checks/
│   │   │   ├── data_quality.py
│   │   │   ├── drift.py
│   │   │   └── performance.py
│   │   │
│   │   ├── alerts/
│   │   │   ├── slack.py
│   │   │   └── email.py
│   │   │
│   │   └── run_monitoring.py
│   │
│   ├── common/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   ├── io.py
│   │   └── utils.py
│
├── pipelines/
│   ├── train_pipeline.py
│   ├── inference_pipeline.py
│   └── retrain_pipeline.py
│
├── webapp/
│   ├── api/
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── predictor.py
│   │
│   ├── ui/
│   │   ├── templates/
│   │   │   └── index.html
│   │   └── static/
│   │       └── style.css
│   │
│   └── run.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── data/
│
├── scripts/
│   ├── run_training.sh
│   ├── run_inference.sh
│   └── run_monitoring.sh
│
├── docker/
│   ├── training.Dockerfile
│   └── serving.Dockerfile
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── pyproject.toml
├── Makefile
└── README.md
```
---