import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def fetch_stock_data(ticker="2638.HK", period="5y"):
    """
    Fetch historical stock data from Yahoo Finance
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def prepare_features(df):
    """
    Create technical indicators as features
    """
    # Basic price features
    df['Returns'] = df['Close'].pct_change()
    df['Price_Momentum'] = df['Close'] - df['Close'].shift(5)
    
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Volume features
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # Price ranges
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = df['Daily_Range'] / df['Close']
    
    # Trend indicators
    df['Price_Position'] = (df['Close'] - df['MA20']) / df['MA20']
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data_for_training(df, target_days=5):
    """
    Prepare data for training, including target variable
    """
    # Create target variable (future price after target_days)
    df['Target'] = df['Close'].shift(-target_days)
    
    # Drop rows with NaN targets
    df = df.dropna()
    
    # Select features for training
    features = ['Close', 'Returns', 'Price_Momentum', 'MA5', 'MA20', 'MA50',
               'Volatility', 'Volume_MA5', 'Volume_MA20', 'Daily_Range_Pct',
               'Price_Position', 'RSI']
    
    # Create feature matrix and target vector
    X = df[features].copy()
    y = df['Target'].copy()
    
    return X, y

def train_xgboost_model(X, y):
    """
    Train XGBoost model with optimized parameters
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features while preserving feature names
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Initialize and train XGBoost model with optimized parameters
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, X_test_scaled, y_test, y_pred, mse, r2, feature_importance

def plot_results(y_test, y_pred, feature_importance):
    """
    Plot actual vs predicted prices and feature importance
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Actual vs Predicted Prices
    ax1.plot(y_test.index, y_test.values, label='Actual Price', linewidth=2)
    ax1.plot(y_test.index, y_pred, label='XGBoost Predictions', linestyle='--', alpha=0.7)
    ax1.set_title('2638.HK Stock Price Prediction (XGBoost)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (HKD)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Feature Importance
    feature_importance.plot(x='feature', y='importance', kind='bar', ax=ax2)
    ax2.set_title('Feature Importance in XGBoost Model')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('xgboost_analysis.png')
    plt.close()

def predict_future_prices(model, scaler, last_data, feature_names, days_to_predict=252):
    """
    Predict future prices using the trained XGBoost model with dynamic feature updates
    """
    future_predictions = []
    current_data = pd.DataFrame([last_data], columns=feature_names)
    
    # Calculate initial volatility
    initial_volatility = current_data.iloc[0]['Volatility']
    
    for i in range(days_to_predict):
        # Make prediction
        next_pred = model.predict(current_data)[0]
        future_predictions.append(next_pred)
        
        # Update Close price
        prev_close = current_data.iloc[0]['Close']
        current_data.iloc[0, current_data.columns.get_loc('Close')] = next_pred
        
        # Update Returns
        returns = (next_pred - prev_close) / prev_close
        current_data.iloc[0, current_data.columns.get_loc('Returns')] = returns
        
        # Update Price Momentum
        current_data.iloc[0, current_data.columns.get_loc('Price_Momentum')] = next_pred - prev_close
        
        # Update Moving Averages
        current_data.iloc[0, current_data.columns.get_loc('MA5')] = (current_data.iloc[0]['MA5'] * 4 + next_pred) / 5
        current_data.iloc[0, current_data.columns.get_loc('MA20')] = (current_data.iloc[0]['MA20'] * 19 + next_pred) / 20
        current_data.iloc[0, current_data.columns.get_loc('MA50')] = (current_data.iloc[0]['MA50'] * 49 + next_pred) / 50
        
        # Update Volatility (using exponential decay)
        decay_factor = 0.95
        new_volatility = initial_volatility * (decay_factor ** (i/20))  # Decay over time
        current_data.iloc[0, current_data.columns.get_loc('Volatility')] = new_volatility
        
        # Update Volume MAs (assuming slight random variation)
        vol_change = np.random.normal(1, 0.1)  # Random volume multiplier
        current_data.iloc[0, current_data.columns.get_loc('Volume_MA5')] *= vol_change
        current_data.iloc[0, current_data.columns.get_loc('Volume_MA20')] *= vol_change
        
        # Update Daily Range
        range_pct = abs(np.random.normal(0, new_volatility))
        current_data.iloc[0, current_data.columns.get_loc('Daily_Range_Pct')] = range_pct
        
        # Update Price Position
        current_data.iloc[0, current_data.columns.get_loc('Price_Position')] = (next_pred - current_data.iloc[0]['MA20']) / current_data.iloc[0]['MA20']
        
        # Update RSI (simplified)
        current_rsi = current_data.iloc[0]['RSI']
        if returns > 0:
            new_rsi = current_rsi + (1 - current_rsi/100) * abs(returns) * 100
        else:
            new_rsi = current_rsi - (current_rsi/100) * abs(returns) * 100
        current_data.iloc[0, current_data.columns.get_loc('RSI')] = np.clip(new_rsi, 0, 100)
    
    return future_predictions

def plot_future_prediction(y_test, y_pred, future_dates, future_predictions):
    """
    Plot historical and future predicted prices
    """
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='blue', linewidth=2)
    plt.plot(y_test.index, y_pred, label='Historical Predictions', color='green', alpha=0.7)
    
    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Predictions', 
             color='red', linestyle='--', alpha=0.7)
    
    plt.title('2638.HK Stock Price Prediction with XGBoost (Including 2025 Forecast)')
    plt.xlabel('Date')
    plt.ylabel('Price (HKD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('xgboost_future_prediction.png')
    plt.close()

def main():
    # Fetch data
    print("Fetching stock data...")
    df = fetch_stock_data()
    
    # Prepare features
    print("Preparing features...")
    df = prepare_features(df)
    
    # Prepare data for training
    print("Preparing data for training...")
    X, y = prepare_data_for_training(df)
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model, scaler, X_test_scaled, y_test, y_pred, mse, r2, feature_importance = train_xgboost_model(X, y)
    
    # Print metrics
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Plot results and feature importance
    print("\nPlotting analysis results...")
    plot_results(y_test, y_pred, feature_importance)
    
    # Predict future prices
    print("\nPredicting prices up to 2026...")
    last_date = y_test.index[-1]
    end_date = pd.Timestamp('2026-01-01').tz_localize(last_date.tz)
    days_to_2026 = pd.date_range(start=last_date, end=end_date, freq='B')
    future_dates = days_to_2026[1:]  # Exclude the last known date
    future_predictions = predict_future_prices(model, scaler, X_test_scaled.iloc[-1].values, X_test_scaled.columns, len(future_dates))
    
    # Print predicted prices for 2025
    print("\nPredicted monthly prices for 2025:")
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_predictions})
    future_df.set_index('Date', inplace=True)
    monthly_predictions = future_df['Predicted_Price'].resample('ME').last()
    print(monthly_predictions[monthly_predictions.index.year == 2025])
    
    # Plot future predictions
    print("\nPlotting future predictions...")
    plot_future_prediction(y_test, y_pred, future_dates, future_predictions)
    
    print("\nAnalysis completed! Check:")
    print("1. 'xgboost_analysis.png' for model performance and feature importance")
    print("2. 'xgboost_future_prediction.png' for future price predictions")

if __name__ == "__main__":
    main()
