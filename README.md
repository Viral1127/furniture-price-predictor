# 🪑 Furniture Price Predictor

A machine learning-powered web application that estimates the **future resale value of furniture** based on its attributes like purchase year, material, quality, and original price. Built using Python and Streamlit, the app also visualizes year-wise depreciation using ML predictions — not fixed depreciation formulas.

## 🚀 Features

- 📅 Predict future furniture prices by selecting target year  
- 🧠 Trained **Random Forest Regressor** model on realistic data  
- 📉 ML-based depreciation graph showing yearly value drop  
- ✅ Smart input validation (e.g., prevents invalid years)  
- 💰 Clearly highlighted predicted price and depreciation percentage  
- 🌐 Deployed on **Streamlit Cloud** for public access and sharing  

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Backend/ML**: Python, scikit-learn, pandas, numpy  
- **IDE**: VS Code  

## 📂 Project Structure

furniture-price-predictor/<br>
├── app.py # Main Streamlit app <br>
├── furniture_price_dataset.csv # Dataset used for training <br>
└── requirements.txt # Python dependencies


## 📦 Installation

To run this project locally:

```bash
git clone https://github.com/Viral1127/furniture-price-predictor
cd furniture-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live Demo

Deployed on Streamlit Cloud:  
🔗 [Click here to open app](https://furniture-price-predictor-bnrkjkgwmbwuvc9w5kbzo3.streamlit.app/)
