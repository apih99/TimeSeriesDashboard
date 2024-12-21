# Time Series Forecasting Web Application

An interactive web application for time series forecasting using multiple models (ARIMA, SARIMA, and Prophet). Built with Streamlit, this application allows users to upload their time series data and compare different forecasting models.

## Features

- 📊 Support for multiple time series forecasting models:
  - ARIMA (Autoregressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - Prophet (Facebook's forecasting tool)
- 📈 Interactive visualizations of forecasts and residuals
- 📉 Comprehensive performance metrics:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - R² Score
  - MAPE (Mean Absolute Percentage Error)
- 🔄 Model comparison and best model selection
- 📥 Example data generation and download
- 🎯 Adjustable training data size
- 📋 Detailed data information and statistics

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run arima_forecast.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Upload your CSV file or use the example data provided

4. Select the columns you want to analyze

5. Choose a forecasting model and adjust the training size

6. Click "Run Forecast" to see the results

## Data Format

Your CSV file should contain:
- Numeric columns for forecasting
- Each row represents a time point
- No missing values in the selected columns

Example format:
```csv
date,value1,value2,value3
2023-01-01,100,200,150
2023-01-02,102,205,155
2023-01-03,98,195,145
```

## Performance Metrics

The application provides several metrics to evaluate model performance:
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R²**: R-squared score (higher is better, max 1.0)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

## Models

1. **ARIMA**
   - Best for stationary data
   - No seasonal patterns
   - Basic time series forecasting

2. **SARIMA**
   - Handles seasonal patterns
   - More complex than ARIMA
   - Good for data with regular patterns

3. **Prophet**
   - Handles missing data well
   - Automatically detects seasonality
   - Robust to outliers
   - Good for business forecasting

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 