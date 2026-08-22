# Production ML Trading

A production-oriented machine learning project for building, testing, and eventually deploying a systematic trading strategy using financial market data.

## Project Goal

The goal is to build an end-to-end ML trading system while learning production machine learning practices.

The project will cover:

- Market data ingestion
- Data cleaning
- Feature engineering
- ML target creation
- Time-series model training
- Model evaluation
- Backtesting
- Testing and version control
- Production pipelines
- Model monitoring and retraining

## Current Pipeline

Yahoo Finance  
↓  
Raw Market Data  
↓  
Data Cleaning  
↓  
Feature Engineering  
↓  
ML Target  
↓  
Model Training  
↓  
Trading Signals  
↓  
Backtesting  
↓  
Production Pipeline

## Project Structure

```text
production-ml-trading/
│
├── backtests/          # Trading strategy backtests
├── data/               # Local market data (not committed to Git)
├── models/             # Saved ML models
├── notebooks/          # Exploratory analysis
├── src/                # Production Python code
├── tests/              # Automated tests
├── requirements.txt
├── .gitignore
└── README.md
