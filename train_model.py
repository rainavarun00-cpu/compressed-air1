import sqlite3
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest


DATABASE = "data/sensor_data.db"


# Read SQLite
conn = sqlite3.connect(DATABASE)

df = pd.read_sql_query(
    "SELECT * FROM sensor_data",
    conn
)

conn.close()


if len(df) < 30:

    print("Not enough data.")
    print("Let sensor_simulator.py run for some time.")
    print("Then run this file again.")

    exit()


features = [
    "pressure_bar",
    "flow_lpm",
    "temperature_c",
    "power_kw"
]


X = df[features]


model = IsolationForest(
    n_estimators=150,
    contamination=0.15,
    random_state=42
)


model.fit(X)


joblib.dump(
    model,
    "air_leak_model.pkl"
)


print("================================")
print("AI MODEL TRAINING COMPLETE")
print("================================")

print("Training samples:", len(df))

print("Model saved as:")
print("air_leak_model.pkl")