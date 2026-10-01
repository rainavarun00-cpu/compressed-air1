import json
import random
import time
import ssl
import sqlite3
import os
import threading

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import paho.mqtt.client as mqtt


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE = "data/sensor_data.db"
MODEL = "air_leak_model.pkl"

# ------------------------------------------------------------
# HiveMQ configuration
# ------------------------------------------------------------
# You can also replace these values directly with your
# HiveMQ Cloud credentials.

BROKER = "e8ea9428f2c240499e7a7dadf2dbbef3.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "Varun Raina"
PASSWORD = "Varan#111"
TOPIC = "compressed_air/sensor"


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Compressed Air Energy Loss Detection",
    page_icon="💨",
    layout="wide"
)


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database():

    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(
        DATABASE,
        timeout=10,
        check_same_thread=False
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,

            pressure_bar REAL,

            flow_lpm REAL,

            temperature_c REAL,

            power_kw REAL

        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# SAVE SENSOR DATA
# ============================================================

def save_sensor_data(data):

    try:

        conn = sqlite3.connect(
            DATABASE,
            timeout=10,
            check_same_thread=False
        )

        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sensor_data
            (
                timestamp,
                pressure_bar,
                flow_lpm,
                temperature_c,
                power_kw
            )
            VALUES (?, ?, ?, ?, ?)
        """, (

            data["timestamp"],
            data["pressure_bar"],
            data["flow_lpm"],
            data["temperature_c"],
            data["power_kw"]

        ))

        conn.commit()
        conn.close()

    except Exception as e:

        print("SQLite error:", e)


# ============================================================
# GENERATE SENSOR DATA
# ============================================================

def generate_sensor_data():

    # Normal operating condition

    pressure = random.uniform(
        6.5,
        8.0
    )

    flow = random.uniform(
        35,
        55
    )

    temperature = random.uniform(
        25,
        35
    )

    power = random.uniform(
        4.0,
        7.0
    )


    # 15% chance of simulated leak

    leak = random.random() < 0.15


    if leak:

        flow = random.uniform(
            70,
            100
        )

        pressure = random.uniform(
            5.0,
            6.3
        )

        power = random.uniform(
            7.0,
            10.0
        )


    data = {

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "pressure_bar":
            round(
                pressure,
                2
            ),

        "flow_lpm":
            round(
                flow,
                2
            ),

        "temperature_c":
            round(
                temperature,
                2
            ),

        "power_kw":
            round(
                power,
                2
            )
    }


    return data


# ============================================================
# MQTT PUBLISHER
# ============================================================

mqtt_client = None


def start_mqtt():

    global mqtt_client

    if mqtt_client is not None:
        return


    try:

        mqtt_client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="compressed_air_sensor"
        )

        mqtt_client.username_pw_set(
            USERNAME,
            PASSWORD
        )

        mqtt_client.tls_set(
            cert_reqs=ssl.CERT_REQUIRED
        )


        print("Connecting to HiveMQ...")


        mqtt_client.connect(
            BROKER,
            PORT,
            keepalive=60
        )


        print("Connected to HiveMQ")

        mqtt_client.loop_start()


        # Start publishing in background

        def publisher():

            while True:

                try:

                    data = generate_sensor_data()

                    payload = json.dumps(
                        data
                    )


                    mqtt_client.publish(
                        TOPIC,
                        payload
                    )


                    # Also save directly to SQLite

                    save_sensor_data(
                        data
                    )


                    print(
                        "Published:",
                        data
                    )


                except Exception as e:

                    print(
                        "Publisher error:",
                        e
                    )


                time.sleep(2)


        thread = threading.Thread(
            target=publisher,
            daemon=True
        )

        thread.start()


    except Exception as e:

        print(
            "MQTT connection error:",
            e
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(
        DATABASE
    ):

        return pd.DataFrame()


    try:

        conn = sqlite3.connect(
            DATABASE,
            timeout=10
        )


        df = pd.read_sql_query(
            """
            SELECT *
            FROM sensor_data
            ORDER BY id DESC
            LIMIT 100
            """,
            conn
        )


        conn.close()


        if not df.empty:

            df = df.sort_values(
                "id"
            )


        return df


    except Exception as e:

        st.error(
            f"Database error: {e}"
        )

        return pd.DataFrame()


# ============================================================
# LOAD AI MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(
        MODEL
    ):

        return None


    try:

        return joblib.load(
            MODEL
        )

    except Exception as e:

        st.warning(
            f"AI model error: {e}"
        )

        return None


# ============================================================
# START DATABASE
# ============================================================

create_database()


# ============================================================
# START MQTT
# ============================================================

if "mqtt_started" not in st.session_state:

    start_mqtt()

    st.session_state.mqtt_started = True


# ============================================================
# TITLE
# ============================================================

st.title(
    "💨 Compressed-Air Energy Loss Detection System"
)

st.caption(
    "MQTT → HiveMQ → SQLite → AI/ML → Dashboard → Alert"
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# CHECK DATA
# ============================================================

if df.empty:

    st.warning(
        "⚠️ Waiting for sensor data..."
    )

    st.info(
        "The application is waiting for MQTT sensor readings."
    )

    time.sleep(2)

    st.rerun()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "timestamp",
    "pressure_bar",
    "flow_lpm",
    "temperature_c",
    "power_kw"

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    st.error(
        f"Missing database columns: {missing_columns}"
    )

    st.stop()


# ============================================================
# CONVERT DATA TYPES
# ============================================================

numeric_columns = [

    "pressure_bar",
    "flow_lpm",
    "temperature_c",
    "power_kw"

]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(
    subset=numeric_columns
)


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# ============================================================
# AI DETECTION
# ============================================================

status = "UNKNOWN"

score = 0.0


model = load_model()


if model is not None:

    try:

        features = [

            "pressure_bar",
            "flow_lpm",
            "temperature_c",
            "power_kw"

        ]


        X = df[features]


        predictions = model.predict(
            X
        )


        if hasattr(
            model,
            "decision_function"
        ):

            scores = model.decision_function(
                X
            )

        else:

            scores = [
                0
                for _ in range(len(df))
            ]


        latest_prediction = predictions[-1]

        latest_score = scores[-1]


        score = round(
            float(latest_score),
            4
        )


        if latest_prediction == -1:

            status = "POSSIBLE AIR LEAK"

        else:

            status = "NORMAL"


    except Exception as e:

        status = "UNKNOWN"

        st.warning(
            f"AI model error: {e}"
        )


else:

    st.warning(
        "⚠️ AI model not found: "
        "air_leak_model.pkl"
    )


# ============================================================
# LATEST SENSOR DATA
# ============================================================

latest = df.iloc[-1]


# ============================================================
# SENSOR METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Pressure",
        f"{latest['pressure_bar']:.2f} bar"
    )


with col2:

    st.metric(
        "Air Flow",
        f"{latest['flow_lpm']:.2f} L/min"
    )


with col3:

    st.metric(
        "Temperature",
        f"{latest['temperature_c']:.2f} °C"
    )


with col4:

    st.metric(
        "Power",
        f"{latest['power_kw']:.2f} kW"
    )


with col5:

    st.metric(
        "AI Status",
        status
    )


st.divider()


# ============================================================
# AI ANALYSIS
# ============================================================

st.subheader(
    "🤖 AI Analysis"
)


st.metric(
    "Anomaly Score",
    score
)


# ============================================================
# ALERT
# ============================================================

if status == "POSSIBLE AIR LEAK":

    st.error(
        "🚨 ALERT: Possible compressed-air "
        "leakage detected!"
    )

elif status == "NORMAL":

    st.success(
        "✅ System operating normally."
    )

else:

    st.warning(
        "⚠️ AI status unavailable."
    )


# ============================================================
# PRESSURE CHART
# ============================================================

st.subheader(
    "📊 Pressure"
)


fig_pressure = go.Figure()


fig_pressure.add_trace(
    go.Scatter(

        x=df["timestamp"],

        y=df["pressure_bar"],

        mode="lines+markers",

        name="Pressure"

    )
)


fig_pressure.update_layout(

    xaxis_title="Time",

    yaxis_title="Pressure (bar)",

    height=400

)


st.plotly_chart(
    fig_pressure,
    use_container_width=True
)


# ============================================================
# AIR FLOW CHART
# ============================================================

st.subheader(
    "📊 Air Flow"
)


fig_flow = go.Figure()


fig_flow.add_trace(
    go.Scatter(

        x=df["timestamp"],

        y=df["flow_lpm"],

        mode="lines+markers",

        name="Air Flow"

    )
)


fig_flow.update_layout(

    xaxis_title="Time",

    yaxis_title="Flow (L/min)",

    height=400

)


st.plotly_chart(
    fig_flow,
    use_container_width=True
)


# ============================================================
# TEMPERATURE / POWER
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# TEMPERATURE
# ------------------------------------------------------------

with col1:

    st.subheader(
        "🌡️ Temperature"
    )


    fig_temp = go.Figure()


    fig_temp.add_trace(
        go.Scatter(

            x=df["timestamp"],

            y=df["temperature_c"],

            mode="lines",

            name="Temperature"

        )
    )


    fig_temp.update_layout(

        xaxis_title="Time",

        yaxis_title="Temperature (°C)",

        height=350

    )


    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )


# ------------------------------------------------------------
# POWER
# ------------------------------------------------------------

with col2:

    st.subheader(
        "⚡ Power Consumption"
    )


    fig_power = go.Figure()


    fig_power.add_trace(
        go.Scatter(

            x=df["timestamp"],

            y=df["power_kw"],

            mode="lines",

            name="Power"

        )
    )


    fig_power.update_layout(

        xaxis_title="Time",

        yaxis_title="Power (kW)",

        height=350

    )


    st.plotly_chart(
        fig_power,
        use_container_width=True
    )


# ============================================================
# LIVE DATA TABLE
# ============================================================

st.subheader(
    "📡 Latest Sensor Data"
)


display_columns = [

    "timestamp",
    "pressure_bar",
    "flow_lpm",
    "temperature_c",
    "power_kw"

]


st.dataframe(

    df[display_columns].tail(20),

    use_container_width=True,

    hide_index=True

)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.subheader(
    "🔗 System Information"
)


info1, info2, info3, info4 = st.columns(4)


info1.metric(
    "Database",
    "SQLite"
)


info2.metric(
    "Data Source",
    "HiveMQ / MQTT"
)


info3.metric(
    "AI Model",
    "Air Leak Detection"
)


info4.metric(
    "Update Interval",
    "2 seconds"
)


# ============================================================
# LAST UPDATE
# ============================================================

st.caption(
    f"Last sensor reading: "
    f"{latest['timestamp']}"
)


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(2)

st.rerun()
