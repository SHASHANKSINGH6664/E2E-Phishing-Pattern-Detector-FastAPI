# 🛡️ E2E Phishing Pattern Detector — FastAPI

An end-to-end machine learning system that detects phishing URLs by analyzing their **lexical and structural patterns** — no live crawling, no external threat-intel API calls, no page rendering. A URL is broken down into 18 numeric features (length, entropy, special-character ratios, obfuscation signals, TLD trust score, etc.), fed into a trained classifier, and scored as **phishing** or **legitimate** in real time through a FastAPI backend with a lightweight web UI.

**🔗 Live demo:** `http://13.62.160.66:8000` *(deployed on AWS EC2 — see [Deployment](#-deployment-aws-ec2--elastic-ip) below)*

---

## 📌 Table of Contents

- [How It Works — Project Workflow](#-how-it-works--project-workflow)
- [System Design](#-system-design)
- [Feature Engineering](#-feature-engineering)
- [Model Training](#-model-training)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Running Locally](#-running-locally)
- [Running with Docker](#-running-with-docker)
- [Deployment (AWS EC2 + Elastic IP)](#-deployment-aws-ec2--elastic-ip)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)

---

## 🔄 How It Works — Project Workflow

The project is split into two independent phases: an **offline training pipeline** that produces a model artifact, and an **online inference service** that serves predictions using that artifact.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     PHASE 1 — OFFLINE TRAINING                       │
│                                                                      │
│   data.csv ──▶ load_dataset() ──▶ build_lookup_tables()             │
│                                          │                           │
│                                          ▼                           │
│                              extract_features() per URL              │
│                                          │                           │
│                                          ▼                           │
│                       Feature Matrix (20 numeric columns)            │
│                                          │                           │
│                     ┌────────────────────┼────────────────────┐      │
│                     ▼                    ▼                    ▼      │
│           Logistic Regression     Random Forest           XGBoost    │
│              (+ StandardScaler)   (GridSearchCV)         (weighted)  │
│                     │                    │                    │      │
│                     └────────────────────┼────────────────────┘      │
│                                          ▼                           │
│                       Evaluate all 3 on held-out test set            │
│                          (Accuracy, F1-score)                        │
│                                          │                           │
│                                          ▼                           │
│                    Select the BEST model by F1-score                 │
│                                          │                           │
│                                          ▼                           │
│           model.pkl  (model + scaler + lookup tables + feature list) │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      PHASE 2 — ONLINE INFERENCE                      │
│                                                                      │
│   Browser / Client                                                   │
│        │  GET /            (loads static/index.html UI)              │
│        │  POST /predict {"url": "..."}                               │
│        ▼                                                             │
│   FastAPI app (app/main.py)                                          │
│        │  1. Load model.pkl once at startup (lifespan hook)          │
│        │  2. extract_features(url, lookup_tables)                    │
│        │  3. Apply scaler (if the winning model needs one)           │
│        │  4. model.predict() + model.predict_proba()                 │
│        ▼                                                             │
│   JSON Response: { status, probability, model_used, features }       │
│        │                                                             │
│        ▼                                                             │
│   UI renders a phishing / legitimate verdict + feature breakdown     │
└──────────────────────────────────────────────────────────────────────┘
```

**In short:**
1. `app/train.py` reads the labeled dataset, engineers 20 features per URL, trains three candidate models, benchmarks them, and pickles the best-performing one (model + scaler + lookup tables) into `model.pkl`.
2. `app/main.py` loads that `model.pkl` **once** when the FastAPI server starts (not per-request), exposes a `/predict` endpoint, and serves a static HTML/JS frontend for interactive testing.
3. Because feature extraction is purely lexical/structural (regex, string ops, `urlparse`), a prediction is returned in milliseconds with **no network calls to the target URL** — making the service fast, safe (it never visits the potentially malicious link), and easy to containerize.

---

## 🏗️ System Design

```
                              ┌───────────────────────────┐
                              │        End User           │
                              │  (Browser / curl / client)│
                              └──────────────┬────────────┘
                                             │ HTTP (port 8000)
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │        AWS EC2 Instance                   │
                       │        (Elastic IP attached)              │
                       │                                           │
                       │   ┌───────────────────────────────────┐   │
                       │   │  Docker Container                 │   │
                       │   │  ┌─────────────────────────────┐  │   │
                       │   │  │   Uvicorn ASGI Server       │  │   │
                       │   │  │   (app.main:app)            │  │   │
                       │   │  │                             │  │   │
                       │   │  │  ┌───────────────────────┐  │  │   │
                       │   │  │  │  FastAPI Application  │  │  │   │
                       │   │  │  │                       │  │  │   │
                       │   │  │  │ GET /        → UI     │  │  │   │
                       │   │  │  │ GET /health  → status │  │  │   │
                       │   │  │  │ POST/predict → verdict│  │  │   │
                       │   │  │  │ GET /static/*→ assets │  │  │   │
                       │   │  │  └──────────┬────────────┘  │  │   │
                       │   │  │             │               │  │   │
                       │   │  │             ▼               │  │   │
                       │   │  │  ┌───────────────────────┐  │  │   │
                       │   │  │  │ In-memory model_bundle│  │  │   │
                       │   │  │  │  (loaded at startup   │  │  │   │
                       │   │  │  │   from model.pkl)     │  │  │   │
                       │   │  │  │  - model              │  │  │   │
                       │   │  │  │  - scaler (optional)  │  │  │   │
                       │   │  │  │  - lookup_tables      │  │  │   │
                       │   │  │  │  - feature_names      │  │  │   │
                       │   │  │  └───────────────────────┘  │  │   │
                       │   │  └─────────────────────────────┘  │   │
                       │   └───────────────────────────────────┘   │
                       └───────────────────────────────────────────┘
```

### Design decisions

| Decision | Reasoning |
|---|---|
| **Lexical features only (no live crawling)** | The URL is never fetched, so the service is instant, stateless, and cannot itself become an attack vector (no SSRF risk from visiting attacker-controlled links). |
| **Model + scaler + lookup tables bundled in one `model.pkl`** | Everything the inference layer needs (`joblib.load`) is self-contained — no separate config files to keep in sync with the model. |
| **Model loaded once via FastAPI `lifespan`** | Avoids the cost of deserializing the model on every request; the model lives in memory for the life of the process. |
| **Best-of-3 model selection (Logistic Regression / Random Forest / XGBoost)** | `train.py` trains all three and automatically keeps the one with the highest F1-score, so the deployed model is empirically the strongest for the current dataset. |
| **Stateless container, single service** | The whole app (API + static UI) ships as one Docker image, which keeps the EC2 deployment to a single container exposing one port — simple to run, restart, and reverse-proxy later if needed. |
| **`/health` endpoint** | Lets you verify the container and model are both up before wiring in a load balancer, uptime monitor, or CI/CD health check. |

---

## 🧪 Feature Engineering

`app/features.py` converts a raw URL string into **20 numeric features**, purely from its structure — no page content is ever fetched:

| Category | Features |
|---|---|
| **Length-based** | `url_length`, `domain_length`, `tld_length` |
| **Structural** | `is_domain_ip`, `subdomain_count`, `equals_count`, `qmark_count`, `ampersand_count`, `other_special_char_count`, `special_char_ratio` |
| **Obfuscation** | `has_obfuscation`, `obfuscation_ratio` (detects `%XX` hex-encoded characters) |
| **Composition** | `letter_ratio_url`, `digit_ratio_url`, `char_continuation_rate` (how often consecutive characters share the same class: letter/digit/symbol) |
| **Trust signals** | `tld_legitimate_prob` (learned from training data — how often each TLD appears on legitimate sites)|
| **Randomness** | `url_entropy` (Shannon entropy — phishing URLs are often more "random-looking") |
| **Heuristic** | `has_sensitive_keyword_or_shortener` (flags keywords like `login`, `verify`, `secure`, `banking`, and known URL shorteners like `bit.ly`, `tinyurl.com`) |

The `tld_legitimate_prob` and `url_char_prob` lookup tables are **learned from the training set only** (in `build_lookup_tables`) and then saved inside `model.pkl`, so inference at request-time is a fast dictionary lookup rather than a fresh computation.


Normalization: The system explicitly strips http://, https://, and www. from the URL prior to extraction to prevent the model from overfitting to standard web prefixes and skewing length or subdomain metrics.
---

## 🤖 Model Training

Run via:
```bash
python -m app.train --data data.csv --out model.pkl
```

What happens:
1. **Load & normalize labels** — accepts either the `(url, status)` schema or the `(URL, label)` schema (see `load_dataset`).
2. **80/20 stratified train/test split**.
3. **Build lookup tables** (`tld_legit_prob`, `char_prob`) from the training split only, to avoid leakage.
4. **Feature extraction** for every URL using `extract_features`.
5. **Train three candidates:**
   - `LogisticRegression` (with `StandardScaler`, `class_weight="balanced"`)
   - `RandomForestClassifier` (hyperparameter search via `GridSearchCV`)
   - `XGBClassifier` (with `scale_pos_weight` tuned to class imbalance)
6. **Evaluate** each on the held-out test set (accuracy, F1-score, classification report).
7. **Select the winner** — the model with the highest F1-score.
8. **Persist** the winning `model`, its `scaler` (if any), `lookup_tables`, and `feature_names` together into `model.pkl` via `joblib`.
9. **Save metrics** for all three models to `metrics.json` for comparison/auditing.

Optional flags:
- `--sample N` — subsample the dataset to N rows for a quick local test run.
- `--metrics-out path.json` — customize where evaluation metrics are written.

---

## 📁 Project Structure

```
E2E-Phishing-Pattern-Detector-FastAPI/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app: routes, model loading, prediction endpoint
│   ├── features.py       # URL → 20-feature vector extraction logic
│   ├── train.py          # Offline training pipeline (3 models, auto-select best)
│   └── schemas.py        # Pydantic request/response models
├── static/
│   └── index.html         # Single-page UI to test URLs interactively
├── data_augmented.csv     # Training dataset (labeled phishing/legitimate URLs)
├── model.pkl              # Trained model bundle (model + scaler + lookup tables)
├── Dockerfile              # Container build for deployment
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📡 API Reference

### `GET /`
Serves the static HTML frontend (`static/index.html`).

### `GET /health`
Health check — confirms the service is up and the model is loaded.
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
Classifies a single URL.

**Request:**
```json
{ "url": "http://secure-appleid-update.tk/login" }
```

**Response:**
```json
{
  "url": "http://secure-appleid-update.tk/login",
  "status": "phishing",
  "probability": 0.9421,
  "model_used": "random_forest",
  "features": {
    "url_length": 39,
    "domain_length": 25,
    "is_domain_ip": 0,
    "...": "..."
  }
}
```

If `model.pkl` isn't present when the server starts, `/predict` returns **HTTP 503** until a model is trained and placed alongside the app.

Interactive Swagger docs are auto-generated by FastAPI at **`/docs`**, and ReDoc at **`/redoc`**.

---

## 💻 Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/SHASHANKSINGH6664/E2E-Phishing-Pattern-Detector-FastAPI.git
cd E2E-Phishing-Pattern-Detector-FastAPI

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain the model — model.pkl is already included in the repo
python -m app.train --data data_augmented.csv --out model.pkl

# 5. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** for the UI, or **http://localhost:8000/docs** for the API explorer.

---

## 🐳 Running with Docker

```bash
# Build the image
docker build -t phishing-detector .

# Run the container
docker run -d -p 8000:8000 --name phishing-detector phishing-detector
```

Visit **http://localhost:8000**.

---

## ☁️ Deployment (AWS EC2 + Elastic IP)

This project is deployed on an **AWS EC2** instance with a static **Elastic IP** attached, so the service is reachable at a fixed public address instead of an IP that changes on instance restart.

**High-level steps used for this deployment:**

1. **Launch an EC2 instance** (Ubuntu, e.g. `t2.micro` / `t3.small`) in the desired region.
2. **Allocate and associate an Elastic IP** to the instance, so the public IP stays fixed across reboots/stops.
3. **Open inbound ports** in the instance's Security Group:
   - `22` (SSH) — restricted to your IP
   - `8000` (or `80` if reverse-proxied) — for HTTP access to the API/UI
4. **Install Docker** on the instance:
   ```bash
   sudo apt update && sudo apt install -y docker.io
   sudo systemctl enable --now docker
   ```
5. **Clone the repo and build the image** on the instance:
   ```bash
   git clone https://github.com/SHASHANKSINGH6664/E2E-Phishing-Pattern-Detector-FastAPI.git
   cd E2E-Phishing-Pattern-Detector-FastAPI
   sudo docker build -t phishing-detector .
   ```
6. **Run the container** (optionally with `--restart unless-stopped` for resilience across reboots):
   ```bash
   sudo docker run -d -p 8000:8000 --restart unless-stopped --name phishing-detector phishing-detector
   ```
7. **Access the app** via the Elastic IP:
   ```
   http://13.62.160.66:8000
   ```

> 🔗 **Live app:** `http://13.62.160.66:8000` — replace with the actual Elastic IP once deployed.

**Optional hardening for production:**
- Put **Nginx** or an **AWS Application Load Balancer** in front of the container to serve on port `80`/`443` with TLS.
- Add a **domain name** (Route 53) pointing to the Elastic IP instead of exposing the raw IP.
- Set up a **systemd** service or Docker restart policy so the container survives instance reboots.
- Add basic **rate limiting** in front of `/predict` to prevent abuse of the public endpoint.

---

## 🧰 Tech Stack

- **API Framework:** FastAPI + Uvicorn
- **ML/Data:** scikit-learn (Logistic Regression, Random Forest), XGBoost, pandas, NumPy
- **Serialization:** joblib
- **Validation:** Pydantic
- **Frontend:** Vanilla HTML/CSS/JS (served as a static file)
- **Containerization:** Docker
- **Deployment:** AWS EC2 + Elastic IP

---

## 🚀 Future Improvements

- Add CI/CD (GitHub Actions) to auto-build and push the Docker image on merge to `main`.
- Add a reverse proxy (Nginx) + HTTPS via Let's Encrypt in front of the EC2 instance.
- Persist prediction logs for monitoring model drift over time.
- Add batch prediction endpoint (`POST /predict/batch`) for scanning multiple URLs at once.
- Periodic retraining pipeline as new labeled data becomes available.
