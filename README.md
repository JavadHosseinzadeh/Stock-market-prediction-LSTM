# Stock-market-prediction-LSTM

This is a repository of MATLAB scripts that makes use of the LSTM neural network to predict the price of gold. In this project, LSTM models were trained using historical stock market data like gold prices. Predictions of future closing prices based on different technical indicators will be developed.

## Overview of Project
The project has used historical data and technical indicators such as Moving Average, MACD, Bollinger Bands, and Stochastic Oscillator to predict the gold prices. Multiple LSTM models of varied architecture are implemented and trained for future price point prediction. These architectures are further enhanced by using various input variables including the price and technical indicators to present improved predictions.

## Key Files
- **`main.m`**: This is the main script that extracts stock data, computes indicators, normalizes data, and trains the LSTM model. Plots are then used to visualize the actual vs. predicted prices.
- **`LstmIndicatorClose.m`**: Another version of the LSTM model, but this time it uses close prices and a variety of computed indicators as input to run the predictions of gold prices.
- **`LSTMClose.m`**: A simplified LSTM model, this one with inputs from only the closing prices of gold for the predictions.
- **`TenOne.m`** and **`OneOne.m`**: Other versions of the LSTM architecture, where the number of hidden layers and units are varied.
- **`TenTen.m`**: Employes another type of LSTM architecture, together with some technical indicators to make a prediction on the closing price and also assess the model performance.

## Data Extraction and Preprocessing
Stock data is extracted through the `hist_stock_data` function, fetching the historical gold prices between January 2003 and July 2021. The following technical indicators are calculated:
- **Moving Average**
- **MACD (Moving Average Convergence Divergence)**
- **Bollinger Bands**
- **Stochastic Oscillator**

Data is normalized to improve training and performance.

## LSTM Neural Network
There are a couple of versions of the LSTM neural network architectures in this project with different configurations, namely:
- Input layers to accept features such as close price and indicators
- Multiple LSTM layers with dropout layers to regularize against overfitting
- A fully connected layer and regression layer at the end to predict prices

Training is carried out by the Adam optimizer with piecewise learning rate scheduling while results are considered in plots comparing actual vs. predicted prices.
