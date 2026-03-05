# 🌍 TiVA Time-Machine — Nowcasting Global Value Chains

## Background

**Trade in Value Added (TiVA)** is an economic indicator that measures the domestic value added embodied in foreign final demand. Unlike traditional trade statistics that measure gross flows, TiVA reveals where value is **actually created**.

> Example: a smartphone assembled in China may incorporate components from Japan and design from the United States — TiVA captures the American and Japanese value added, while export statistics only record China's shipment.

TiVA data is published by the **OECD** and used by governments, international organizations, and businesses to analyze global value chains. However, it has a critical limitation: **a 3–4 year publication lag**.

---

## 🎯 Challenge Objective

Your mission is to **nowcast** (predict in real time) TiVA flows from **G7 countries** to **80+ world economies**, using only macroeconomic indicators available without delay.

**Target variable**: `TiVA_Value_Target` — domestic value added absorbed by foreign final demand (in millions USD), from the OECD indicator *Foreign Final Demand - Domestic Value Added (FFD_DVA)*.

---

## 📊 Data

The dataset combines two sources:

- **OECD TiVA**: value added flows between G7 source countries and destination countries, by sector and year
- **World Bank**: annual macroeconomic indicators (GDP, inflation, FDI, trade openness, technology, employment…)

**Structure**:
- 7 source countries (G7): CAN, DEU, FRA, GBR, ITA, JPN, USA
- 80+ destination countries
- 72 sectors (ISIC Rev. 4 classification)
- Period: 2005–2020

**Temporal splits**:

| Split | Years | Role |
|---|---|---|
| Train | 2005–2015 | Full access to targets |
| Public test | 2016–2018 | Visible leaderboard |
| Private test | 2019–2020 | Final evaluation |

> ⚠️ The private period (2019–2020) covers the US-China trade war and COVID-19 — two shocks with no equivalent in the training period. Your model's robustness will be tested.

---

## 📐 Evaluation Metric

```
Final Score = 0.3 × MAE_public(2016-2018) + 0.7 × MAE_private(2019-2020)
```

MAE is computed on flows **aggregated by (Year, Source Country)**, not at the individual row level. This evaluates your ability to capture macro-level trends rather than sectoral noise.

**Random Forest baseline (no tuning): ~45,097** *(lower is better)*

---

## 🏁 Getting Started

1. Download the starting kit (Jupyter notebook)
2. Explore the training data
3. Submit a `submission.py` file with a `get_model()` function
4. Iterate using leaderboard feedback

See the **Data**, **Evaluation**, and **Starting Kit** pages for more details.