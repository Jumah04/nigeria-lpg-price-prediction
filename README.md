# Multi-Target LPG Price Prediction & Market Intelligence System

> **Predictive Analytics and Multi-Output Machine Learning for Energy Pricing in Nigeria**  
> *Capstone Project — 3MTT / DeepTech Ready Program (Data Science Nigeria × WesOnline)*  
> **Author:** Jumah Abdulmojeed | **Mentor:** Israel Odeajo | **Date:** August 2026

---

## 📌 Project Overview & Problem Statement

Liquefied Petroleum Gas (LPG), commonly known as cooking gas, is an essential energy commodity for Nigerian households and businesses. Over recent years, retail LPG prices have experienced extreme volatility driven by foreign exchange depreciation, international crude oil benchmarks, domestic petrol (PMS) price adjustments, supply chain disruptions from localized conflicts, and festive demand surges.

Consumers struggle to budget for refills, distributors face cash-flow challenges, and policymakers lack granular predictive intelligence. 

This project solves this challenge by developing a **multi-target machine learning system** that simultaneously forecasts retail LPG prices across three granularity levels:
1. **Per Kilogram** (`lpg_price_per_kg`)
2. **5kg Cylinder Refill** (`5kg_price`)
3. **12.5kg Cylinder Refill** (`12.5kg_price`)

The model covers all **36 Nigerian states and the Federal Capital Territory (FCT)** using macroeconomic indicators, temporal factors, and geospatial context.

---

## 📊 Dataset Description

The dataset spans monthly state-level observations from **January 2023 through 2024** (1,370 total observations).

### Raw Feature Inventory
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `state` | Categorical | 37 administrative divisions (36 states + FCT) |
| `region` | Categorical | Geopolitical zone (North Central, North East, North West, South East, South South, South West) |
| `year`, `month` | Integer | Calendar temporal indices |
| `5kg_price` | Float (Target) | Average retail price for a 5kg cylinder (₦) |
| `12.5kg_price` | Float (Target) | Average retail price for a 12.5kg cylinder (₦) |
| `pms_price` | Float | Average Premium Motor Spirit (petrol) retail price (₦/litre) |
| `crude_oil_price`| Float | International Brent crude oil price ($/barrel) |
| `buying_rate` | Float | Daily average Naira-to-Dollar buying exchange rate |
| `selling_rate` | Float | Daily average Naira-to-Dollar selling exchange rate |
| `inflation_rate`| Float | Headline monthly inflation rate (%) |
| `conflict` | Binary (Yes/No)| Presence of active localized conflict in the state-month |
| `festive_period`| Binary (Yes/No)| High-demand cultural and national festive season indicator |

---

## 🛠️ Feature Engineering & Preprocessing

1. **Standardized Per-Kg Target (`lpg_price_per_kg`):**
   $$\text{lpg\_price\_per\_kg} = \frac{1}{2} \left( \frac{\text{5kg\_price}}{5} + \frac{\text{12.5kg\_price}}{12.5} \right)$$
   Averaging unit prices accounts for economies of scale where 12.5kg cylinders typically offer slightly lower per-kg prices.
2. **Unified FX Metric (`dollar_price`):**
   $$\text{dollar\_price} = \frac{\text{buying\_rate} + \text{selling\_rate}}{2}$$
   Combines buying and selling exchange rates to eliminate severe multicollinearity.
3. **Categorical Encoding:**
   - **Label Encoding:** Applied to binary indicators (`conflict`, `festive_period`: `No` $\rightarrow 0$, `Yes` $\rightarrow 1$).
   - **One-Hot Encoding:** Applied to `state` and `region` with `drop_first=True` to prevent the dummy variable trap.

---

## 🔬 Modeling & Experimental Results

### 1. Algorithm Benchmarking (5-Fold Cross-Validation)
The dataset was split into 80% training (1,096 rows) and 20% hold-out testing (274 rows, `random_state=42`). Three model families were evaluated using 5-Fold Cross-Validation:

| Algorithm | Multi-Output Strategy | 5-Fold CV Avg RMSE (₦) | Ranking |
| :--- | :--- | :---: | :---: |
| **Random Forest Regressor** | Native Multi-Output | **522.60** | **1st (Best) 🏆** |
| **XGBoost Regressor** | `MultiOutputRegressor` wrapper | 522.95 | 2nd |
| **Linear Regression** | Native Multi-Output | 1,081.36 | 3rd |

*Random Forest marginally outperformed XGBoost while drastically outperforming Linear Regression, confirming strong non-linear relationships and interactions among macroeconomic drivers.*

---

### 2. Geographical Feature Ablation Study
Rather than applying arbitrary feature pruning that could render the model blind to less populous states, a systematic ablation study was conducted across four spatial variants:

| Variant | Spatial Features Included | Test Set Avg $R^2$ | Outcome |
| :--- | :--- | :---: | :--- |
| **Variant 1 (All features)** | **State dummies + Region dummies** | **0.9488** | **Winner 🏆 (Highest explained variance)** |
| **Variant 4** | Region dummies only (No State) | 0.9393 | Moderate zonal resolution |
| **Variant 3** | State dummies only (No Region) | 0.9391 | High state granularity |
| **Variant 2 (No Geo)** | Macroeconomic & temporal features only | 0.9205 | Lowest accuracy |

*Key Takeaway: Retaining geospatial identity provides an immediate ~2.8 percentage point increase in $R^2$, proving that state-specific supply chains and retail dynamics significantly impact pricing.*

---

### 3. Final Model Evaluation on Held-Out Test Set (Variant 1)

| Target Variable | Test $R^2$ Score | Test RMSE (₦) |
| :--- | :---: | :---: |
| **`lpg_price_per_kg`** | **0.9555** | **₦65.33** |
| **`5kg_price`** | **0.9385** | **₦364.01** |
| **`12.5kg_price`** | **0.9525** | **₦901.72** |
| **Overall Average** | **0.9488** | — |

---

## 💡 Key Market Insights & Findings

1. **FX Depreciation is the Dominant Factor (~67% Feature Importance):** The Naira-to-Dollar exchange rate accounts for roughly two-thirds of predictive power, reflecting Nigeria's high reliance on imported refined LPG.
2. **Fuel Price Spillover (PMS):** PMS (petrol) price is the second largest driver, reflecting inland transportation and trucking logistics costs.
3. **Local Demand vs. Global Supply:** Festive periods trigger significant price spikes locally due to household cooking gas demand, while international crude benchmarks remain unimpacted.
4. **Geopolitical & Conflict Risk Premiums:** Regional security disruptions systematically inflate retail gas prices due to supply routes and depot bottlenecks.

---

## 🚀 Interactive Streamlit Web Application

The project includes an inference deployment pipeline:
- Serializes the trained Random Forest model (`lpg_model.pkl`) and feature alignment metadata (`model_metadata.pkl`) using `joblib`.
- Built for an interactive **Streamlit** user interface allowing consumers, marketers, and analysts to select their State, geopolitical region, current FX rates, and PMS prices to get instant real-time LPG price predictions.

---

## 📁 Repository Structure

```text
nigeria-lpg-price-prediction/
├── Jumah Abdulmojeed Project Notebook.ipynb   # Complete ML pipeline, EDA, ablation & evaluation
├── Jumah LPG Dataset.csv                      # Primary dataset (1,370 records)
├── lpg_prediction_data.csv                    # Dataset alias matching notebook input
├── .gitignore                                 # Git ignore configuration
└── README.md                                  # Comprehensive documentation & project report
```

---

## 💻 Getting Started

### Prerequisites
Install the required Python packages:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap joblib streamlit
```

### Running the Notebook
Launch Jupyter Notebook to view and reproduce the entire pipeline:
```bash
jupyter notebook "Jumah Abdulmojeed Project Notebook.ipynb"
```

---

## 👥 Acknowledgments
- **Mentor:** Israel Odeajo
- **Program Sponsors:** 3MTT Nigeria, Data Science Nigeria (DSN), WesOnline
