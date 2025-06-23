import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("furniture_price_dataset.csv")

df["Original_Price"] = df["Original_Price"].replace("[₹,]", "", regex=True).astype(int)
df["Predicted_Price"] = df["Predicted_Price"].replace("[₹,]", "", regex=True).astype(int)

label_encoders = {}
categorical_cols = ['Furniture_Type', 'Material', 'Quality', 'Color']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[['Purchase_Year', 'Original_Price', 'Furniture_Type', 'Material', 'Quality', 'Color', 'Condition_Years', 'Target_Year']]
y = df['Predicted_Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.set_page_config(page_title="Furniture Price Predictor", page_icon="🪑")
st.title("🪑 Furniture Price Predictor")
st.write("Estimate the resale value of your furniture in the future based on its age, material, and quality.")

st.header("📝 Enter Furniture Details")
purchase_year = st.selectbox("Purchase Year", sorted(df['Purchase_Year'].unique()))
target_year = st.number_input("Target Year", min_value=2024, max_value=2035, value=2026, step=1)
original_price = st.number_input("Original Price (₹)", min_value=1000, max_value=100000, value=25000, step=1000)

furniture_type = st.selectbox("Furniture Type", label_encoders['Furniture_Type'].classes_)
material = st.selectbox("Material", label_encoders['Material'].classes_)
quality = st.selectbox("Quality", label_encoders['Quality'].classes_)
color = st.selectbox("Color", label_encoders['Color'].classes_)

condition_years = target_year - purchase_year

if condition_years < 0:
    st.error("⚠️ Target year must be after purchase year!")
else:
    input_data = pd.DataFrame([[purchase_year,
                                original_price,
                                label_encoders['Furniture_Type'].transform([furniture_type])[0],
                                label_encoders['Material'].transform([material])[0],
                                label_encoders['Quality'].transform([quality])[0],
                                label_encoders['Color'].transform([color])[0],
                                condition_years,
                                target_year]],
                              columns=X.columns)

    st.markdown("---")
    st.subheader("🔮 Predicted Output")

    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price in {target_year}: ₹{int(predicted_price):,}")

    st.subheader("📉 ML-Based Depreciation Over Time")

    ml_years = list(range(purchase_year, target_year + 1))
    ml_prices = []

    for year in ml_years:
        if year == purchase_year:
            ml_prices.append(original_price)
        else:
            condition = year - purchase_year
            input_row = pd.DataFrame([[
                purchase_year,
                original_price,
                label_encoders['Furniture_Type'].transform([furniture_type])[0],
                label_encoders['Material'].transform([material])[0],
                label_encoders['Quality'].transform([quality])[0],
                label_encoders['Color'].transform([color])[0],
                condition,
                year
            ]], columns=X.columns)

            price = model.predict(input_row)[0]
            ml_prices.append(price)

    chart_df = pd.DataFrame({
        "Year": ml_years,
        "ML Predicted Value (₹)": [int(p) for p in ml_prices]
    })

    st.line_chart(chart_df.set_index("Year"))

    initial = ml_prices[0]
    final = ml_prices[-1]
    percent_drop = round((initial - final) / initial * 100, 2)
    st.info(f"🔻 Estimated ML-Based Depreciation from ₹{int(initial):,} to ₹{int(final):,} is **{percent_drop}%**")
