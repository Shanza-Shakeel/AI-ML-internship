import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure app
st.set_page_config(
    page_title="Stock Predictor Pro",
    page_icon="📈",
    layout="wide"
)

# App title
st.title("📈 Stock Price Prediction Pro")
st.write("Predict next day's closing price with machine learning")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL, TSLA)", "TSLA").strip().upper()
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())
model_type = st.sidebar.selectbox("Model Type", ["Linear Regression", "Random Forest"])

# Fixed data loading function
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error("No data found for this ticker")
            return None
        
        # PROPERLY handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]
            
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main function
def main():
    data = load_data(ticker, start_date, end_date)
    if data is None:
        st.warning("Please check your inputs and try again")
        return
    
    # Show raw data
    st.subheader(f"Historical Data for {ticker}")
    st.dataframe(data.tail())
    
    # Plot historical prices
    st.subheader("Price Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['close'], color='blue', linewidth=2)
    ax.set_title(f"{ticker} Closing Prices", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)
    
    # Prepare data
    df = data.copy()
    df['next_close'] = df['close'].shift(-1)
    df = df.dropna()
    
    features = ['open', 'high', 'low', 'close', 'volume']
    X = df[features]
    y = df['next_close']
    
    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_scaled, y)
    
    # Prediction section
    st.subheader("Make Prediction")
    
    if st.button("Predict Tomorrow's Price"):
        try:
            last_data = X_scaled[-1].reshape(1, -1)
            prediction = model.predict(last_data)[0]
            current_price = df['close'].iloc[-1]
            change_pct = ((prediction - current_price) / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Predicted Price", f"${prediction:.2f}")
            col3.metric("Predicted Change", f"{change_pct:.2f}%")
            
            st.success("Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    # Model evaluation
    st.subheader("Model Performance")
    predictions = model.predict(X_scaled)
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y, predictions)):.2f}")
    st.write(f"R² Score: {r2_score(y, predictions):.2f}")
    
    # Feature importance (for Random Forest)
    if model_type == "Random Forest":
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis', ax=ax2)
        ax2.set_title("Feature Importance Scores")
        st.pyplot(fig2)

if __name__ == "__main__":
    main()