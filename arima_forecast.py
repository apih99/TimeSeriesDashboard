import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import streamlit as st

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("Interactive Time Series Forecasting")

# Instructions
with st.expander("📖 Instructions & Information", expanded=True):
    st.markdown("""
    ### How to Use This App
    1. **Upload Data**: Upload a CSV file containing your time series data
        - Data should be numeric
        - Each column represents a different variable
        - Rows represent time points
    
    2. **Select Columns**: Choose which numeric columns to analyze
        - Multiple columns will be combined by taking their mean
    
    3. **Choose Model**:
        - **ARIMA**: Good for stationary data with no seasonal patterns
        - **SARIMA**: Better for data with seasonal patterns
        - **Prophet**: Handles seasonality and missing values well
        - **All Models**: Compare all three models
    
    4. **Adjust Training Size**: Select what percentage of data to use for training
    
    ### Example Data Format
    Your CSV should look like this:
    ```
    date,value1,value2,value3
    2023-01-01,100,200,150
    2023-01-02,102,205,155
    2023-01-03,98,195,145
    ```
    
    ### Performance Metrics
    - **RMSE**: Root Mean Square Error (lower is better)
    - **MAE**: Mean Absolute Error (lower is better)
    - **R²**: R-squared score (higher is better, max 1.0)
    - **MAPE**: Mean Absolute Percentage Error (lower is better)
    """)

# Example Data Download
with st.expander("📥 Download Example Data"):
    # Generate example data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    example_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(100, 15, 100) + np.sin(np.arange(100)/10)*10,
        'temperature': np.random.normal(25, 5, 100) + np.sin(np.arange(100)/10)*3,
        'visitors': np.random.normal(500, 50, 100) + np.sin(np.arange(100)/10)*50
    })
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(example_data)
    st.download_button(
        "Download Example CSV",
        csv,
        "example_timeseries.csv",
        "text/csv",
        key='download-example-csv'
    )
    st.dataframe(example_data.head())

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Data Info
    with st.expander("📊 Data Information"):
        st.write("Dataset Shape:", df.shape)
        st.write("Column Types:")
        st.write(df.dtypes)
        st.write("Missing Values:")
        st.write(df.isnull().sum())
    
    # Column selection
    st.subheader("Select Columns for Analysis")
    # Filter only numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns found in the dataset. Please ensure your CSV contains numeric data for forecasting.")
    else:
        selected_columns = st.multiselect(
            "Choose numeric columns for forecasting",
            options=numeric_columns,
            default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
        )
        
        if selected_columns:
            # Get the data for selected columns
            data = df[selected_columns].copy()
            
            # Convert data to numeric, replacing errors with NaN
            for col in selected_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            if len(data) == 0:
                st.error("No valid numeric data found in selected columns after cleaning.")
            else:
                # Combine selected columns into one time series by taking the mean
                combined_series = data.mean(axis=1)
                
                # Model selection
                st.subheader("Select Forecasting Model")
                model_choice = st.selectbox(
                    "Choose a forecasting model",
                    ["ARIMA", "SARIMA", "Prophet", "All Models"]
                )
                
                # Training size selection
                train_size = st.slider(
                    "Select training data size (%)",
                    min_value=50,
                    max_value=90,
                    value=80,
                    step=5
                )

def calculate_metrics(actual, predictions):
    """Calculate multiple performance metrics."""
    rmse = sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    }

def plot_predictions(train, test, predictions, model_name, metrics):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Plot the main forecasting graph
    ax1.plot(train.index, train, label='Training Data', color='blue')
    ax1.plot(test.index, test, label='Actual Test Data', color='green')
    ax1.plot(test.index, predictions, label='Predictions', color='red', linestyle='--')
    ax1.set_title(f'{model_name} Forecast')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Combined Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the residuals
    residuals = test - predictions
    ax2.plot(test.index, residuals, color='gray', label='Residuals')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals (Actual - Predicted)')
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Residual Value')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Function to perform ARIMA forecasting
def perform_arima_forecast(series, train_size_pct):
    train_size = int(len(series) * train_size_pct / 100)
    train, test = series[:train_size], series[train_size:]
    
    with st.spinner('Running ARIMA forecast...'):
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))
        metrics = calculate_metrics(test, predictions)
        
        # Display metrics
        st.write("### ARIMA Performance Metrics")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df)
        
        fig = plot_predictions(train, test, predictions, 'ARIMA', metrics)
        st.pyplot(fig)
        
    return predictions, metrics

# Function to perform SARIMA forecasting
def perform_sarima_forecast(series, train_size_pct):
    train_size = int(len(series) * train_size_pct / 100)
    train, test = series[:train_size], series[train_size:]
    
    with st.spinner('Running SARIMA forecast...'):
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        predictions = model_fit.forecast(steps=len(test))
        metrics = calculate_metrics(test, predictions)
        
        # Display metrics
        st.write("### SARIMA Performance Metrics")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df)
        
        fig = plot_predictions(train, test, predictions, 'SARIMA', metrics)
        st.pyplot(fig)
        
    return predictions, metrics

# Function to perform Prophet forecasting
def perform_prophet_forecast(series, train_size_pct):
    df_prophet = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=len(series), freq='H'),
        'y': series
    })
    
    train_size = int(len(series) * train_size_pct / 100)
    train = df_prophet.iloc[:train_size]
    test = df_prophet.iloc[train_size:]
    
    with st.spinner('Running Prophet forecast...'):
        model = Prophet(yearly_seasonality=True, 
                       weekly_seasonality=True, 
                       daily_seasonality=True,
                       changepoint_prior_scale=0.05)
        model.fit(train)
        
        future = model.make_future_dataframe(periods=len(test), freq='H')
        forecast = model.predict(future)
        
        predictions = forecast.iloc[train_size:]['yhat'].values
        metrics = calculate_metrics(test['y'], predictions)
        
        # Display metrics
        st.write("### Prophet Performance Metrics")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df)
        
        fig = plot_predictions(train['y'], test['y'], predictions, 'Prophet', metrics)
        st.pyplot(fig)
        
    return predictions, metrics

if uploaded_file is not None and selected_columns:
    if st.button("Run Forecast"):
        st.subheader("Forecasting Results")
        
        results = {}
        
        if model_choice in ["ARIMA", "All Models"]:
            arima_predictions, arima_metrics = perform_arima_forecast(combined_series, train_size)
            results["ARIMA"] = arima_metrics
            
        if model_choice in ["SARIMA", "All Models"]:
            sarima_predictions, sarima_metrics = perform_sarima_forecast(combined_series, train_size)
            results["SARIMA"] = sarima_metrics
            
        if model_choice in ["Prophet", "All Models"]:
            prophet_predictions, prophet_metrics = perform_prophet_forecast(combined_series, train_size)
            results["Prophet"] = prophet_metrics
        
        if len(results) > 1:
            st.subheader("Model Comparison")
            # Create a comparison dataframe with all metrics
            comparison_data = {}
            for model_name, metrics in results.items():
                comparison_data[model_name] = metrics
            
            comparison_df = pd.DataFrame(comparison_data).T
            st.dataframe(comparison_df)
            
            # Find best model for each metric
            best_models = {}
            for metric in ['RMSE', 'MAE', 'MAPE (%)', 'R²']:
                if metric != 'R²':
                    best_model = comparison_df[metric].idxmin()
                    best_value = comparison_df[metric].min()
                else:
                    best_model = comparison_df[metric].idxmax()
                    best_value = comparison_df[metric].max()
                best_models[metric] = (best_model, best_value)
            
            # Display best models
            st.subheader("Best Models by Metric")
            for metric, (model, value) in best_models.items():
                st.success(f"Best model for {metric}: {model} ({value:.2f})")