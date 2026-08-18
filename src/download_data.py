import yfinance as yf

data = yf.download("AAPL", period="1mo")

data.to_csv("data/AAPL_raw.csv")

print("Saved AAPL data to data/AAPL_raw.csv")
print("Shape:", data.shape)

