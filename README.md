# Retail Performance & Forecasting Dashboard

This project is a comprehensive **Sales Forecasting and Analytics Dashboard** designed for large-scale retail data (based on Walmart datasets). It helps business owners understand historical trends, identify top-performing stores, and predict future sales using advanced AI and Time-Series models.

## How to Run
1. Ensure you have the dependencies installed (`streamlit`, `pandas`, `statsmodels`, `matplotlib`, `scikit-learn`).
2. Run the dashboard using:
   ```bash
   streamlit run app.py
   ```

##  Dashboard Screenshots Explained

I have included several images to help you understand the different sections of the dashboard:

### 1. **User Interaction: Store & Department Selection** 
To get started, the dashboard allows you to select exactly what you want to analyze:
*   **Store Selection:** ![Store Selection](store_selection.png) Choose from 45 different store locations.
*   **Department Selection:** ![Department Selection](department_selection.png) Drill down into specific categories like Grocery, Electronics, or Apparel.

### 2. **Key Performance Indicators (KPIs)** ![KPI Dashboard](kpi_dashboard.png)
This is the top row of the dashboard. It shows:
*   **Total Store Revenue:** The total sales for the selected store.
*   **Stationery Avg:** The average weekly sales for specific categories (like Stationery).
*   **Holiday Sales Lift:** A powerful metric showing how many **times more** you sell during holidays compared to normal weeks (e.g., 1.1x lift).

### 3. **Raw Data & Granular Tracking** ![Raw Data Tracking](raw_data_tracking.png)
This shows the "back-end" of the data. 
*   It displays every week's performance alongside external factors like **Temperature**, **Fuel Price**, and **Unemployment**.
*   This help you see exactly what was happening in the world when a specific sale occurred.

### 4. **AI Forecasting (SARIMA vs ARIMA)** ![AI Forecasting](ai_forecasting.png)
This is the most "intelligent" part of the project:
*   **Historical (Blue line):** Your actual past sales.
*   **ARIMA (Dashed line):** A basic forecast that only looks at recent trends (ends up being low because it misses the "big picture").
*   **SARIMA (Red line):** Our advanced **Seasonal AI**. It "remembers" that sales spike every December and predicts those peaks for the future.
*   **Smart Predictions:** On the right, it shows direct dollar-value predictions using **Random Forest** (complex patterns) and **Linear Regression** (simple trends).

### 5. **Regional & Department Rankings** ![Regional Rankings](regional_rankings.png)
This provides a "bird's-eye view" of your entire business:
*   **Top Stores (Left):** Identifies which of your 45+ locations are bringing in the most revenue.
*   **Top Departments (Right):** Shows which categories—like **Grocery** or **Deli**—are your volume leaders.

## 🧠 Key Technologies
*   **Python:** Core data processing.
*   **Streamlit:** Interactive web interface.
*   **SARIMA:** Seasonal time-series forecasting for holiday-aware predictions.
*   **ARIMA:** Auto-Regressive Integrated Moving Average for baseline trend forecasting.
*   **Random Forest:** Machine learning for complex pattern recognition (Temperature, Fuel, etc.).
*   **Metadata Mapping:** Translates anonymized department IDs (1-99) into real-world names for better understanding.
