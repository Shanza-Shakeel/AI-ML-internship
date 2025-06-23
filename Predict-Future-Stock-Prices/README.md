# Stock Price Prediction App

## 📌 Project Overview
A machine learning application that predicts the next day's closing price for stocks using historical market data.

## 🎯 Task Objective
- Develop a predictive model for short-term stock price movements
- Compare performance of different ML algorithms
- Create an interactive Streamlit web application for predictions

## 📊 Dataset
- **Source**: Yahoo Finance (via `yfinance` API)
- **Tickers**: Any valid stock symbol (AAPL, TSLA, MSFT, etc.)
- **Features**:
  - Open price
  - High price
  - Low price
  - Close price
  - Trading volume
- **Time Period**: User-selectable (default: 3 years historical data)

## 🤖 Models Applied
| Model | Type | Key Characteristics |
|-------|------|---------------------|
| **Linear Regression** | Statistical | Fast training, interpretable coefficients |
| **Random Forest** | Ensemble ML | Handles non-linearity, robust to outliers |

## 🔍 Key Findings
1. **Prediction Accuracy**:
   - Random Forest typically achieves better accuracy (lower RMSE)
   - Linear Regression provides more conservative estimates

2. **Feature Importance** (Random Forest):
   - Current closing price is most significant predictor
   - Trading volume has relatively low importance

3. **Performance Metrics**:
   - Typical R² scores: 0.85-0.95 on test data
   - RMSE varies by stock volatility (usually 2-5% of price)

## 🚀 How to Use
1. Install requirements:
   ```bash
   pip install streamlit yfinance scikit-learn pandas numpy matplotlib seaborn

## How to Run the App
1. Clone this repository:
   ```bash
   git clone https://github.com/Shanza-Shakeel/AI-ML-internship/Predict-Future-Stock-Prices.git 
   cd Predict-Future-Stock-Prices