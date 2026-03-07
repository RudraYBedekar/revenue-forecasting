from data_loader import load_data, merge_datasets
from preprocessing import preprocess_data
from eda_analysis import sales_trend, holiday_analysis
from forecasting_model import train_model, train_linear_regression, train_arima_model, train_sarima_model
from evaluation import evaluate_model, evaluate_arima_model


def main():

    # Load datasets
    train, features, stores = load_data()

    # Merge datasets
    df = merge_datasets(train, features, stores)

    # Preprocess data
    df = preprocess_data(df)

    # Save cleaned dataset for Streamlit dashboard
    df.to_csv("retail_sales_cleaned.csv", index=False)

    # Train and Evaluate Random Forest Model
    rf_model, X_test, y_test = train_model(df)
    evaluate_model("Random Forest", rf_model, X_test, y_test)

    # Train and Evaluate Linear Regression Model
    lr_model, X_test_lr, y_test_lr = train_linear_regression(df)
    evaluate_model("Linear Regression", lr_model, X_test_lr, y_test_lr)

    # Train and Evaluate ARIMA Model
    arima_fit, ts_data = train_arima_model(df)
    evaluate_arima_model(arima_fit, ts_data)

    # Train and Evaluate SARIMA Model
    print("\nTraining SARIMA (This may take a moment)...")
    sarima_fit, ts_data_sarima = train_sarima_model(df)
    evaluate_arima_model(sarima_fit, ts_data_sarima) # Using same eval for now

    # Show analysis graphs
    sales_trend(df)
    holiday_analysis(df)


if __name__ == "__main__":
    main()