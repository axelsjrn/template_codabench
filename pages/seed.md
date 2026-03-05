# Starting Kit

Download the Jupyter notebook `tiva_starting_kit.ipynb` from the **Resources** section. Run it step by step to reproduce the baseline and understand the full pipeline.

---

## Step 1 — Load the data

```python
DATA_DIR = 'dev_phase/input_data/'

X_train = pd.read_csv(DATA_DIR + 'X_train.csv')
X_test  = pd.read_csv(DATA_DIR + 'X_test.csv')
y_train = pd.read_csv(DATA_DIR + 'y_train.csv')
# y_test is not available — scoring is handled by Codabench

df_train = pd.concat([X_train, y_train['TiVA_Value_Target']], axis=1)
df_test  = pd.concat([X_test, pd.Series([0]*len(X_test), name='TiVA_Value_Target')], axis=1)
```

---

## Step 2 — Feature Engineering

The ingestion program applies this exact pipeline — replicate it in your notebook to get consistent scores:

```python
def fill_missing(df):
    macro_cols = [c for c in df.columns if c.endswith('_Source') or c.endswith('_Target')]
    for col in macro_cols:
        group_key = 'Source_Country' if col.endswith('_Source') else 'Target_Country'
        df[col] = (df.groupby(group_key)[col]
                     .transform(lambda x: x.interpolate(method='linear').ffill().bfill()))
    df[macro_cols] = df[macro_cols].fillna(df[macro_cols].median())
    return df

def feature_engineering(df):
    df['GDP_ratio']             = np.log1p(df['GDP_USD_Source']) - np.log1p(df['GDP_USD_Target'])
    df['Tech_gap']              = df['HighTech_Export_Source']   - df['HighTech_Export_Target']
    df['RnD_gap']               = df['Research_Spend_Source']    - df['Research_Spend_Target']
    df['Unemployment_gap']      = df['Unemployment_Source']      - df['Unemployment_Target']
    df['Trade_openness_mean']   = (df['Trade_Openness_Source']   + df['Trade_Openness_Target']) / 2
    df['Trade_openness_diff']   = df['Trade_Openness_Source']    - df['Trade_Openness_Target']
    df['Market_size_target']    = np.log1p(df['GDP_USD_Target']  * df['Population_Target'])
    df['GDP_per_capita_target'] = np.log1p(df['GDP_USD_Target']  / (df['Population_Target'] + 1))
    df['Macro_stability']       = -np.abs(df['Inflation_Target'])
    df['Digital_gap']           = df['Internet_Users_Source']    - df['Internet_Users_Target']
    df['Tech_intensity_source'] = df['HighTech_Export_Source']   * df['Research_Spend_Source']
    df['FDI_ratio']             = np.log1p(df['FDI_Inflow_Source']) / (np.log1p(df['FDI_Inflow_Target']) + 1e-6)
    df['Year_norm']   = (df['Year'] - 2000) / 20
    df['Crisis_2008'] = df['Year'].isin([2008, 2009]).astype(int)
    df = pd.get_dummies(df, columns=['Source_Country', 'Sector_Code'], drop_first=False)
    drop_cols = [c for c in ['Sector_Name','Target_Country','country_x','country_y','Sector_Code_GTN'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

df_train = fill_missing(df_train)
df_test  = fill_missing(df_test)

df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all = feature_engineering(df_all)
```

---

## Step 3 — Split & Normalize

```python
from sklearn.preprocessing import StandardScaler

onehot_cols = [c for c in df_all.columns if c.startswith('Source_Country_')]

train_mask = df_all['Year'] <= 2015
pub_mask   = df_all['Year'].between(2016, 2018)
priv_mask  = df_all['Year'].between(2019, 2020)

X_train = df_all[train_mask].drop(columns=['TiVA_Value_Target', 'Year'])
y_train = df_all[train_mask]['TiVA_Value_Target']
X_pub   = df_all[pub_mask].drop(columns=['TiVA_Value_Target', 'Year'])
y_pub   = df_all[pub_mask]['TiVA_Value_Target']
refs_pub = df_all[pub_mask][onehot_cols + ['Year']]
X_priv  = df_all[priv_mask].drop(columns=['TiVA_Value_Target', 'Year'])
y_priv  = df_all[priv_mask]['TiVA_Value_Target']
refs_priv = df_all[priv_mask][onehot_cols + ['Year']]

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_pub_sc   = pd.DataFrame(scaler.transform(X_pub),   columns=X_pub.columns,   index=X_pub.index)
X_priv_sc  = pd.DataFrame(scaler.transform(X_priv),  columns=X_priv.columns,  index=X_priv.index)
```

---

## Step 4 — Baseline Model

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)

y_pred_pub  = rf.predict(X_pub_sc)
y_pred_priv = rf.predict(X_priv_sc)
```

---

## Step 5 — Official Scoring

```python
from sklearn.metrics import mean_absolute_error

def scoring_func(pred_pub, true_pub, refs_pub, pred_priv, true_priv, refs_priv):
    def _aggregate(pred, true, refs):
        refs = refs.copy()
        refs['Source_Country'] = refs[onehot_cols].idxmax(axis=1).str.replace('Source_Country_', '')
        df = refs[['Year', 'Source_Country']].reset_index(drop=True)
        df['pred'] = pd.Series(pred).values
        df['true'] = pd.Series(true).values
        return df.groupby(['Year', 'Source_Country'])[['pred', 'true']].sum()

    df_pub  = _aggregate(pred_pub,  true_pub,  refs_pub)
    df_priv = _aggregate(pred_priv, true_priv, refs_priv)

    mae_pub  = mean_absolute_error(df_pub['true'],  df_pub['pred'])
    mae_priv = mean_absolute_error(df_priv['true'], df_priv['pred'])
    score    = 0.3 * mae_pub + 0.7 * mae_priv

    print(f'MAE Public  : {mae_pub:>12,.2f}')
    print(f'MAE Privé   : {mae_priv:>12,.2f}')
    print(f'Score Final : {score:>12,.2f}')
    return score, df_pub, df_priv

score, df_pub_agg, df_priv_agg = scoring_func(
    y_pred_pub, y_pub, refs_pub,
    y_pred_priv, y_priv, refs_priv
)
```

---

## Step 6 — Submit

Create `submission.py` with a `get_model()` function and zip it:

```python
# submission.py
def get_model():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
```

```bash
zip submission.zip submission.py
```

Then upload `submission.zip` on Codabench.

---

## Improvement Ideas

- **Log-transform**: predict `log(1 + TiVA)` then invert with `exp(pred) - 1`
- **Gradient Boosting**: XGBoost or LightGBM with hyperparameter tuning
- **Lag features**: previous year's TiVA flows (⚠️ careful about data leakage)
- **Neural networks**: country/sector embeddings