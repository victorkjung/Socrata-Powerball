# Socrata-Powerball
Powerball Analyzer for Entertainment Purposes

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-app-red.svg)

# 📊 NY Powerball Analyzer (Socrata API + Streamlit)

A production-ready **Streamlit web application** that analyzes historical New York Powerball winning numbers using the official **NY Open Data Socrata API**.

The app provides advanced visual analytics, statistical trend scoring, probability simulations, and mobile-optimized dashboards.

**Live dataset source:**  
https://dev.socrata.com/foundry/data.ny.gov/d6yy-54nr  

---

## 🚀 Features

### 🔥 Heat Maps
Monthly frequency heatmaps for:

- White balls  
- Powerball  

---

### 📈 Top & Bottom Numbers

Defined as:

**Top 6 = Top 5 white balls + Top 1 Powerball**  
**Bottom 6 = Bottom 5 white balls + Bottom 1 Powerball**

Includes **monthly win % trend charts**.

---

### 🧠 Hot vs Cold Trend Scoring

Statistical z-score style model comparing:

- Recent draw frequency  
- Long-term baseline  

Highlights:

🔥 Hot numbers rising in frequency  
🧊 Cold numbers dropping in frequency  

---

### 🧮 Probability Simulator

Includes:

- Exact probability for every match pattern  
- Jackpot odds  
- Monte Carlo simulation sessions  

Supports:

- Current Powerball rules (5 from 69 + PB from 26)  
- Dataset-derived number ranges  

---

### ✅ Mock Drawing Checker

Enter:

- 5 white balls (unordered)  
- 1 Powerball  

Searches the full dataset to find:

- Whether the combination ever occurred  
- The exact draw dates  

---

### 🌙 Dark Mode + 📱 Mobile Optimization

- In-app dark/light toggle  
- Responsive layout  
- Compact chart mode  
- Sidebar collapses on mobile  

---

### 🔄 Smart API Caching

Streamlit-native caching with:

- Configurable TTL (1–48 hours)  
- Manual force refresh  
- Rate-limit protection  

---

### 📥 CSV Export

Download:

- Full draw history  
- Long-format exploded dataset  

---

## 📁 Project Structure

```

SocrataPowerball/
│
├── SocrataPowerball.py
├── requirements.txt
├── README.md
├── LICENSE
├── CONTRIBUTING.md
│
├── tests/
│   └── test_api.py
│
└── .github/
└── workflows/
└── ci.yml

````

---

## 📦 Requirements

`requirements.txt`

```txt
streamlit
pandas
numpy
requests
plotly
pytest
````

---

## ⚙️ Local Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Add Socrata API Token (recommended)

Without a token you may hit rate limits.

#### Option A — environment variable:

```bash
export SOCRATA_APP_TOKEN="YOUR_TOKEN"
```

#### Option B — Streamlit secrets:

Create:

```
.streamlit/secrets.toml
```

Add:

```toml
SOCRATA_APP_TOKEN = "YOUR_TOKEN"
```

---

### 3️⃣ Run locally

```bash
streamlit run SocrataPowerball.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push repo to GitHub
2. Visit [https://share.streamlit.io](https://share.streamlit.io)
3. Create new app

**Main file path:**

```
SocrataPowerball.py
```

4. Add token in:

**App → Settings → Secrets**

```toml
SOCRATA_APP_TOKEN = "YOUR_TOKEN"
```

---

## ✅ Continuous Integration

GitHub Actions automatically runs pytest:

* On every push
* On pull requests
* Daily scheduled health check

Validates:

* API availability
* Required dataset fields

---

## 📚 Data Source & Attribution

This application uses publicly available data from:

**New York State Open Data (powered by Socrata)**
Dataset: *Powerball Winning Numbers*
[https://data.ny.gov](https://data.ny.gov)

API endpoint:
[https://data.ny.gov/resource/d6yy-54nr.json](https://data.ny.gov/resource/d6yy-54nr.json)

All data subject to NY Open Data terms of use.

---

## ⚠️ Disclaimer

* Lottery outcomes are independent random events
* Historical trends do NOT predict future results
* Hot/Cold scoring is descriptive only
* Simulator assumes uniform randomness

This project is for **educational and analytical purposes only**.

---

## 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

## 🤝 Contributing

Contributions are welcome!

Please see **CONTRIBUTING.md** for:

* Bug reports
* Feature requests
* Pull request workflow

---

## 🧠 Tech Stack

* Python 3.10+
* Streamlit
* Pandas / NumPy
* Plotly
* Socrata Open Data API
* GitHub Actions CI

---

## ⭐ Future Enhancements

* Daily auto-refresh ping workflow
* Installable PWA shell
* Correlation & streak analysis
* Historical rule change handling
* Alerts for hot/cold number shifts


