import joblib
scaler = joblib.load('models/scaler.pkl')
print(f"Mean: {list(scaler.mean_)}")
print(f"Scale: {list(scaler.scale_)}")
