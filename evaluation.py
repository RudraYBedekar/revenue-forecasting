from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model_name, model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name} Performance")
    print("MAE:", mae)
    print("R2 Score:", r2)

def evaluate_arima_model(model_fit, ts_data):
    # ARIMA evaluation on training data (simplified for this context)
    predictions = model_fit.fittedvalues
    mae = mean_absolute_error(ts_data, predictions)
    
    print("\nARIMA Model Performance (In-sample)")
    print("MAE:", mae)