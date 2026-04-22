# 🔐 PhishGuard – Intelligent Phishing Detection System

PhishGuard is a **multi-stage phishing detection platform** that combines **machine learning, threat intelligence, and real-time browser protection** to identify malicious URLs with high accuracy.

It provides both:

* 🌐 **Web-based analysis dashboard**
* 🧩 **Browser extension for real-time protection**

---

## 🚀 Problem 

Phishing attacks are evolving rapidly, bypassing traditional rule-based detection systems. Users often fall victim due to:

* Lack of real-time warnings
* Sophisticated URL obfuscation
* Delayed blacklist updates

PhishGuard addresses this by combining **ML-based detection + live threat intelligence**.

---

## 💡 Solution Overview

PhishGuard uses a **2-stage machine learning pipeline**:

1. **Stage 1 (Fast Detection)**

   * URL-based feature extraction
   * Lightweight model for quick filtering

2. **Stage 2 (Deep Analysis)**

   * Advanced feature evaluation
   * Behavioral + structural analysis

3. **Threat Intelligence Layer**

   * Integration with phishing datasets (PhishTank/Open feeds)

4. **Final Risk Engine**

   * Outputs: `Safe | Suspicious | Phishing`
   * Confidence-based scoring system

---

## 🧠 Key Features

* 🔍 Real-time URL scanning
* 🧠 ML-based phishing detection (2-stage model)
* ⚡ Fast API using FastAPI
* 🌐 Interactive web dashboard
* 🧩 Chrome extension for instant alerts
* 📊 Risk scoring system (0–100)
* 🛡️ Phishing dataset integration
* 💾 Caching & database support

---

## 🧩 Extension Features

* Detects current tab URL automatically
* One-click scanning
* Instant warning popup
* Blocks suspicious websites
* Lightweight and fast

---

## 🖥️ Tech Stack

### Backend

* Python
* FastAPI
* Scikit-learn
* NumPy / Pandas

### Frontend

* HTML
* CSS
* JavaScript

### ML Pipeline

* Feature Engineering (URL-based)
* Random Forest (Stage 1)
* Advanced Model (Stage 2)

### Database

* SQLite

---

## ⚙️ Project Structure

```
PhishGuard/
│
├── backend/
│   ├── app/
│   ├── ml/
│   ├── core/
│   ├── data/
│   └── run.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/Alwaysparth/PhishGuard.git
cd PhishGuard
```

### 2. Setup virtual environment

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run backend server

```
cd backend
python run.py
```

Server will start at:

```
http://localhost:8000
```

---

## 🌐 API Endpoints

### Check URL

```
POST /check-url
```

**Request**

```json
{
  "url": "https://example.com",
  "mode": "viewer"
}
```

**Response**

```json
{
  "status": "safe",
  "risk_score": 12,
  "confidence": 0.92
}
```

---

## 📊 Workflow

1. User submits URL (Web / Extension)
2. Backend validates request
3. Feature extraction (31+ features)
4. Stage 1 ML classification
5. Stage 2 deep analysis
6. Threat intelligence check
7. Final risk score generated

---

## 📈 Model Evaluation

* Dataset: 20000 samples (balanced)
* Evaluation: 5-Fold Cross Validation
* Metrics: Accuracy, Precision, Recall, F1

> Note: Real-world validation with live datasets is in progress.

---

## 🔮 Future Enhancements

* 🌍 Live API integration (Google Safe Browsing)
* 🧠 Deep learning model upgrade
* 📊 Real-time analytics dashboard
* ☁️ Cloud deployment
* 📱 Mobile support

---

## 👨‍💻 Team

* Parth Yadav – ML Engineer
* Hritik Dua - Backend Developer
* Ananya Maurya - Frontend Developer
* Kashish Gupta - Database Engineer

---

## 📜 License

This project is for educational and hackathon purposes.

---

## ⭐ Acknowledgements

* kaggle phishing sites url dataset
* Open-source ML community

---

## 🚀 Final Note

PhishGuard is not just a project — it’s a **step toward safer browsing**.

> “Detect before damage. Protect before phishing.”
