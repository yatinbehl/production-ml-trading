def calculate_future_price(price, percentage):
    return price * (1 + percentage)


stock = "AAPL"
price = 227.50

new_price = calculate_future_price(price, 0.10)

print("Stock:", stock)
print("Current price:", price)
print("10% higher price:", round(new_price, 2))

