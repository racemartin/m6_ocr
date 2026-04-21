# MLOps Partie 2/2 — Déployez et Monitorez le Modèle de Scoring

> **Prérequis :** Avoir complété la Partie 1/2 ([README.md](README.md)). Les artefacts MLflow (modèle versionné, `preprocessor.pkl`) et la base PostgreSQL doivent être opérationnels.

---

## 🗺️ Vue d'ensemble de l'architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     Production Stack                          │
│                                                               │
│  FastAPI  ──►  MLflow Registry  ──►  Modèle (ONNX/pkl)        │
│     │                                                         │
│     ▼                                                         │
│  Logger JSON  ──►  Evidently (Drift)  ──► Streamlit Dashboard │ 
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## 📦 Services

| Port | Service     | Localisation                             |
|------|-------------|------------------------------------------|
| 5433 | postgres    | `postgres_data:/var/lib/postgresql/data` |
| 8088 | pgadmin     | `http://localhost:8088``                 |
| 5001 | mlflow      | `http://localhost:5001`                  |
| 8081 | API FastAPI | `src/api/main.py`                        |
| 8501 | dashboard   | `monitoring/dashboard.py`                |
| 8502 | simulator   | `monitoring/simulator.py`                |

---

## 🚀 Installation & Lancement local

### 1. Cloner et configurer l'environnement

```bash
git clone https://github.com/racemartin/m6_ocr.git
cd m6_ocr

cp .env.example .env          # Remplir DATABASE_URL, MLFLOW_TRACKING_URI, MODEL_VERSION
uv sync
```

### 2. Démarrer l'infrastructure (PostgreSQL + MLflow)

```bash
docker compose up -d postgres mlflow pgadmin
docker ps                     # Vérifier que les 3 conteneurs sont UP
```

> **Ports exposés :** PostgreSQL → `5433`, MLflow UI → `http://localhost:5001`, pgAdmin → `http://localhost:8088`

### 3. Vérifier que le modèle est bien enregistré dans MLflow

```bash
# Ouvrir http://localhost:5001 → onglet "Models"
# Le modèle "scoring_credit" doit apparaître avec un stage "Production"
```

---

## 🔌 Étape 2 — Lancer l'API FastAPI

```bash
# En local (développement)
make api
# ou bien 
uv run uvicorn src.api.main:app --reload --port 8000

# Accéder à la documentation Swagger
http://localhost:8001/docs
```

**Vérification santé :**

```bash
curl http://localhost:8001/health
# {"statut":"ok",
#   "scorer_pret":true,
#   "model_backend":"onnx",
#   "version_api":"2.0.0",
#   "seuil_decision":0.35
#  }
```

**Test rapide de l'endpoint de prédiction :**

```bash
curl -X 'POST' \
  'http://192.168.1.146:8001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 35,
  "amt_annuity": 25000,
  "amt_credit": 500000,
  "bureau_credit_total": 5,
  "bureau_debt_mean": 1000,
  "cc_balance_mean": 0,
  "cc_drawings_mean": 0,
  "code_gender": "F",
  "education_type": "Higher education",
  "ext_source_1": 0.5,
  "ext_source_2": 0.5,
  "ext_source_3": 0.5,
  "goods_price": 450000,
  "max_dpd": 0,
  "objet_pret": "Unaccompanied",
  "paymnt_delay_mean": 2,
  "paymnt_ratio_mean": 0.1,
  "phone_change_days": 365,
  "pos_months_mean": 12,
  "region_rating": 2,
  "revenu": 50000,
  "type_pret": "Cash loans",
  "type_residence": "House / apartment",
  "years_employed": 10
}'

# Réponse : 
	
Response body
Download
{
  "id_demande": "a253c8d6-4711-470e-bb9d-b5cca19339b3",
  "probabilite_defaut": 0.6635897755622864,
  "decision": "Refusé",
  "score_risque": 0.6635897755622864,
  "latence_ms": 62.09,
  "seuil_utilise": 0.35,
  "explication_shap": [ ... 
```

> ⚠️ Le modèle est chargé **une seule fois au démarrage** (`@app.on_event("startup")`), pas à chaque requête.
---

## Lancer le simulator pour requeter l'API a travers d'une interface web.

```bash
# En local (développement)
make simulate

open http://localhost:8502/
# Realiser plusieurs requetes avec differents parametres pour 
# remplir le fichier predictions.json utlise plus tard dans le dashboard. 
```

## Lancer le script pour realiser l'analyse du Drift du Modele.

```bash
# En local (développement)
uv run python scripts/drift_analysis.py

# Produit : monitoring/reports/drift_report.html  (référence = données d'entraînement Phase 1)
open monitoring/reports/drift_report.html
```

## Lancer le dashboard pour le suivi de la latence et du drift.

```bash
# En local (développement)
make dashboard

open http://localhost:8501/
```
**Contenu du dashboard :**
- Distribution des scores prédits (temps réel vs référence)
- Latence de l'API (P50, P95, P99)
- Taux de refus par tranche horaire
- Alertes drift (seuil Jensen-Shannon > 0.1)


## Profilage de performance du modèle
Analysez et optimisez les performances du modèle

Lancer l'script pour mesurer les perfomances:
• Comparative des Performances: Sklearn vs ONNX
• Test de charge.
```bash
# En local (développement)
uv run python optimization/profile_model.py

# Produit : ls optimization\rapports\
```
* benchmark_onnx_latences.csv
* benchmark_sklearn_latences.csv
* bench_onnx.json
* bench_sklearn.json
* profiling_onnx.txt
* profiling_sklearn.txt

Métriques surveillées :
* latence_min
* latence_p50
* latence_p75
* latence_p95
* latence_p99
* latence_max
* latence_moy


## 🐳 Étape — Build Docker & déploiement

**CI/CD (GitHub Actions) :** chaque push sur `main` déclenche automatiquement :
1. Tests Pytest
2. Build de l'image Docker
3. Déploiement sur Hugging Face Spaces (port `7860`)

```bash
# Suivre le pipeline
# GitHub → Actions → workflow "CI/CD Pipeline"
```



## 🔄 Séquence complète d'exécution (récapitulatif)

```
# 1. Démarrer l'infrastructure de données et de tracking
docker compose up -d postgres mlflow pgadmin

# 2. Lancer l'API FastAPI (le Scorer doit être prêt)
uv run uvicorn src.api.main:app --reload --port 8001

# 3. Lancer le Simulateur pour générer des données de test (Interface Web)
make simulate  # ou uv run streamlit run monitoring/simulator.py --server.port 8502

# 4. Effectuer des tests unitaires et d'intégration
uv run pytest tests/ -v

# 5. Exécuter l'analyse de Drift (compare les prédictions à la référence)
uv run python scripts/drift_analysis.py

# 6. Lancer le Dashboard de monitoring (Latence et Drift)
make dashboard # ou uv run streamlit run src/dashboard/app.py --server.port 8501

# 7. Lancer le profilage de performance (Benchmark Sklearn vs ONNX)
uv run python optimization/profile_model.py
```

---

## 🔒 Variables d'environnement (`.env`)

| Variable              | Description            | Exemple                                          |
|-----------------------|------------------------|--------------------------------------------------|
| `DATABASE_URL`        | PostgreSQL             | `postgresql://user:pwd@localhost:5433/mlops_db`  |
| `MLFLOW_TRACKING_URI` | Serveur MLflow         | `http://localhost:5001`                          |
| `MODEL_NAME`          | Nom du modèle registre | `scoring_credit`                                 |
| `MODEL_STAGE`         | Stage MLflow           | `Production`                                     |
| `DECISION_THRESHOLD`  | Seuil de décision      | `0.35`                                           |

---

## 📚 Références techniques

- [FastAPI](https://fastapi.tiangolo.com/) · [Evidently AI](https://docs.evidentlyai.com/) · [ONNX Runtime](https://onnxruntime.ai/) · [GitHub Actions](https://docs.github.com/en/actions) · [Streamlit](https://docs.streamlit.io/)

---

## 👤 Auteur

**Rafael Cerezo Martín**

* Email: [rafael.cerezo.martin@icloud.com](mailto:rafael.cerezo.martin@icloud.com)
* GitHub: [@racemartin](https://github.com/racemartin)

## 📄 Licence

MIT License - voir le fichier [LICENSE](https://www.google.com/search?q=LICENSE) pour plus de détails.

