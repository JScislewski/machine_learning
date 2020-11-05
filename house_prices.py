import pandas as pd

melbourne_file_path = "input\melbourne-housing-snapshot\melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())