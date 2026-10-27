"""Local demo UI:  streamlit run deployment/app.py"""
import os, sys, json
import pandas as pd
import streamlit as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import predict

st.set_page_config(page_title="Shopzilla CSAT predictor", layout="centered")
st.title("Shopzilla - CSAT risk predictor")
meta = json.load(open(os.path.join(HERE, "meta.json")))
lv = meta["categorical_levels"]

c1, c2 = st.columns(2)
channel = c1.selectbox("Channel", lv["channel_name"]); category = c2.selectbox("Category", lv["category"])
sub = c1.selectbox("Sub-category", lv["Sub-category"]); product = c2.selectbox("Product category", lv["Product_category"])
tenure = c1.selectbox("Agent tenure", lv["Tenure Bucket"]); shift = c2.selectbox("Agent shift", lv["Agent Shift"])
manager = c1.selectbox("Manager", lv["Manager"]); supervisor = c2.selectbox("Supervisor", lv["Supervisor"])
agent = st.text_input("Agent name (optional - unknown agents use the average)", "")
resp = st.slider("Response time (minutes)", 0, 2880, 10)
price = st.number_input("Item price (0 = not an order ticket)", 0, 200000, 0)
hour = st.slider("Hour the issue was reported", 0, 23, 12)

if st.button("Predict"):
    rep = pd.Timestamp.now().normalize() + pd.Timedelta(hours=hour)
    row = pd.DataFrame([{"channel_name": channel, "category": category, "Sub-category": sub, "Product_category": product,
                         "Tenure Bucket": tenure, "Agent Shift": shift, "Manager": manager, "Supervisor": supervisor,
                         "Agent_name": agent or None, "Item_price": price or None,
                         "Issue_reported at": rep.strftime("%d/%m/%Y %H:%M"),
                         "issue_responded": (rep + pd.Timedelta(minutes=resp)).strftime("%d/%m/%Y %H:%M"),
                         "Survey_response_Date": rep.strftime("%d-%b-%y")}])
    out = predict.score_dataframe(row).iloc[0]
    st.metric("Predicted CSAT", int(out["predicted_csat"]), help=f"Expected score {out['expected_csat']}")
    st.metric("Probability customer is dissatisfied (CSAT<=2)", f"{out['p_dissatisfied']:.0%}")
    st.subheader(f"Risk tier: {out['risk_tier']}")
    st.bar_chart(pd.Series({k: out[f"p_csat_{k}"] for k in range(1, 6)}, name="probability"))
