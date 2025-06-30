import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.title("📊 Time Series Forecasting App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


def plot_decomposition(ts, model_type):
    st.subheader("Seasonal Decomposition")
    if model_type == "Multiplicative" and (ts <= 0).any():
        shift_val = abs(ts.min()) + 1
        ts += shift_val
    result = seasonal_decompose(ts, model=model_type.lower(), period=12)
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axs[0].plot(ts, label="Original", color="blue")
    axs[1].plot(result.trend, label="Trend", color="green")
    axs[2].plot(result.seasonal, label="Seasonality", color="orange")
    axs[3].plot(result.resid, label="Residuals", color="red")
    axs[0].set_title("Original Series")
    axs[1].set_title("Trend")
    axs[2].set_title("Seasonal")
    axs[3].set_title("Residual")
    plt.tight_layout()
    st.pyplot(fig)


def display_metrics(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    try:
        mape = np.mean(np.abs((true - pred) / true)) * 100
    except:
        mape = np.inf
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MAPE:** {'∞%' if np.isinf(mape) else f'{mape:.2f}%'}")
    st.write(f"**MSE:** {mse:.2f}")


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Choose preprocessing method
    if {
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    }.issubset(df.columns):
        use_custom = st.radio("Choose input method", ["Auto (Hotel Format)", "Manual"])
    else:
        use_custom = "Manual"

    if use_custom == "Manual":
        date_col = st.selectbox("Select the Date column", df.columns)
        value_col = st.selectbox("Select the Value column", df.columns)

        # Handle duplicates and resample
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col])
        df = df.set_index(date_col)
        df = df.sort_index()
        df = df[[value_col]].groupby(df.index).sum()  # remove duplicate timestamps
        ts = df[value_col].resample("M").sum().fillna(method="ffill")

    else:
        # Hotel bookings format
        if df["arrival_date_month"].dtype == "object":
            df["arrival_date_month"] = pd.to_datetime(
                df["arrival_date_month"], format="%B"
            ).dt.month
        df["arrival_date"] = pd.to_datetime(
            dict(
                year=df["arrival_date_year"],
                month=df["arrival_date_month"],
                day=df["arrival_date_day_of_month"],
            )
        )
        daily = df.groupby("arrival_date").size().reset_index(name="total_bookings")
        daily.set_index("arrival_date", inplace=True)
        monthly = daily.resample("M").sum()
        monthly.index.name = "Month"
        ts = monthly["total_bookings"].fillna(method="ffill")

    st.subheader("Monthly Time Series")
    st.line_chart(ts)

    decomposition_type = st.selectbox(
        "Choose decomposition model", ["Additive", "Multiplicative"]
    )
    plot_decomposition(ts.dropna(), decomposition_type)

    model_choice = st.selectbox(
        "Choose forecasting model", ["ARIMA", "ETS", "Prophet", "LSTM"]
    )
    periods = st.slider("Forecast Horizon (months)", 6, 36, 12)

    ts = ts.dropna()
    train = ts[:-periods]
    test = ts[-periods:]

    st.subheader(f"{model_choice} Forecast")

    if model_choice == "ARIMA":
        model = ARIMA(train, order=(1, 1, 1)).fit()
        forecast = model.forecast(steps=periods)

    elif model_choice == "ETS":
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=6,
            initialization_method="estimated",
        ).fit()
        forecast = model.forecast(periods)

    elif model_choice == "Prophet":
        prophet_df = train.reset_index()
        prophet_df.columns = ["ds", "y"]
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, freq="M")
        forecast_df = model.predict(future)
        forecast = forecast_df.set_index("ds")["yhat"][-periods:]

    elif model_choice == "LSTM":
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train.values.reshape(-1, 1))
        X, y = [], []
        for i in range(12, len(scaled)):
            X.append(scaled[i - 12 : i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, batch_size=8, verbose=0)
        last_input = scaled[-12:].reshape(1, 12, 1)
        forecast = []
        for _ in range(periods):
            pred = model.predict(last_input)[0, 0]
            forecast.append(pred)
            last_input = np.append(last_input[:, 1:, :], [[[pred]]], axis=1)
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    forecast_series = pd.Series(forecast, index=test.index)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train, label="Train")
    ax.plot(test, label="Test", color="orange")
    ax.plot(forecast_series, label="Forecast", color="green")
    ax.set_title(f"{model_choice} Forecast Results")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Evaluation Metrics")
    display_metrics(test, forecast_series)
