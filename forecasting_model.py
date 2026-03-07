from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

def train_model(df):

    features = [
        "Store","Dept","Temperature","Fuel_Price",
        "CPI","Unemployment","IsHoliday","Size"
    ]

    X = df[features]
    y = df["Weekly_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

def train_linear_regression(df):
    features = [
        "Store","Dept","Temperature","Fuel_Price",
        "CPI","Unemployment","IsHoliday","Size"
    ]

    X = df[features]
    y = df["Weekly_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

def train_arima_model(df):
    # ARIMA requires a time series (sales aggregated by date)
    ts_data = df.groupby("Date")["Weekly_Sales"].sum()
    
    # Simple ARIMA implementation
    model = ARIMA(ts_data, order=(1, 1, 1))
    model_fit = model.fit()
    
    return model_fit, ts_data

def train_sarima_model(df):
    # Aggregating sales by date for time-series
    ts_data = df.groupby("Date")["Weekly_Sales"].sum()
    
    # SARIMA implementation with a 52-week seasonal cycle
    # seasonal_order=(P, D, Q, s) where s=52
    model = SARIMAX(
        ts_data, 
        order=(1, 1, 1), 
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    return model_fit, ts_data