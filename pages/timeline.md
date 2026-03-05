# Timeline

## Development Phase (January → June 2026)

| | |
|---|---|
| **Start** | January 1, 2026 |
| **End** | June 1, 2026 |
| **Submissions per day** | 5 |
| **Total submissions** | 100 |

### What to do
- Train your model on **2005–2015** data
- Evaluate on the **public leaderboard** (2016–2018)
- The final score also includes the **private test (2019–2020)**, hidden during the competition

### Test periods

| Period | Years | Weight in final score |
|---|---|---|
| Public test | 2016–2018 | 30% |
| Private test | 2019–2020 | 70% |

> ⚠️ The private period covers the **US-China trade war** and **COVID-19** — two unprecedented shocks. Optimizing only for the public leaderboard may not generalize well to 2019–2020.

---

## Scoring Formula

```
Final Score = 0.3 × MAE_public(2016–2018) + 0.7 × MAE_private(2019–2020)
```

**Baseline (Random Forest, no tuning): ~45,097** — can you beat it?