# REN Energy Forecasting
## Energy Consumption Forecast

Data:  
Interface - Energy Consumption  
Local Generation - PV Modules Energy Generation

## First Part - Fit SARIMAX, Residual Evaluation and  Fit ARCH to SARIMAX Residuals for Volatility Forecast
### 1
- Data Analysis
- Interpolate Missing hours and days
- Add Holidays and Season of Year to the Equation
- Check Trend and Seasonality
- Evaluate Series Stationarity
- Granger Causality (Measure Causality Between 'Interface' and 'Local Generation')

### 2
- Fit SARIMAX
- Evaluate Residuals - Homoskedacity, Normally Distributed, Volatility Cluster?
- Fit ARCH on Heteroskedastic Residuals
- Test SARIMAX Forecast

![Alt text](images/workflow_ts.png?raw=true "Time Series Workflow")

## Second Part - Fit CNN - LSTM based Architectures to Extract Non Linearity on Data
### 1
- Bidirectional LSTM for Interface Forecasting
- Bidirectional LSTM for Volatility Forecasting

### 2
- CNN for Raw Time Series Feature Extraction

### 3
- Sum Interface and Volatility Forecasting

### 4
- Concatenate Sum of Interface and Volatility Forecasting to CNN Extracted Features

### 4
- Fully Connected Layer for Concat Layer Output Interpretation

### 5
- Ouput Layer for Prediction

![Alt text](images/hybrid_model.png?raw=true "CNN - Bidirectional LSTM - Volatility Model")