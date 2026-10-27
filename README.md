# Shopzilla CSAT Prediction with Deep Learning (ANN)

Predicts customer-satisfaction (CSAT 1-5) and the risk of a *dissatisfied* customer (CSAT <= 2) for e-commerce support tickets, using only information available while the ticket is open.

## Repository layout
```
CSAT_EDA_Project.ipynb   # exploratory data analysis (20 charts, recommendations)
CSAT_ANN_Project.ipynb   # full project notebook (EDA, hypothesis tests, features, 3 models, explainability)
data/eCommerce_Customer_support_data.csv
deployment/
  csat_utils.py                   # feature engineering + preprocessing (shared by notebook and serving)
  predict.py                      # CLI / function for scoring new tickets
  app.py                          # Streamlit demo
  csat_model.keras, preprocessor.joblib, meta.json   # created by the notebook
requirements.txt
```

## Run
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute CSAT_ANN_Project.ipynb   # ~15 min on CPU; creates ./deployment artifacts
python deployment/predict.py new_tickets.csv scored.csv
streamlit run deployment/app.py
```

## Key results (hold-out test, 17,182 tickets)
| Model | ROC-AUC (dissatisfied) | QWK | macro-F1 |
|---|---|---|---|
| Always predict 5 | 0.500 | 0.000 | 0.164 |
| Logistic regression | 0.732 | 0.282 | 0.264 |
| Dense ANN | 0.732 | 0.273 | 0.260 |
| Embedding + residual ANN | 0.734 | 0.283 | 0.262 |

The three models are statistically tied. Ticket metadata caps predictability; response time is the dominant driver. Customer remarks are excluded (target leakage).
