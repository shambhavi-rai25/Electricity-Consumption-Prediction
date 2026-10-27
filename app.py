import streamlit as st
import joblib
import pandas as pd
import numpy as np

models = joblib.load("energy_models.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
model_info = joblib.load("model_info.pkl")

st.set_page_config(
    page_title = "Electricity Consumption Prediction",
    layout = "centered"
)

st.title("Electricity Consumption Prediction")

st.subheader("Enter Weather and Time Information")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=20.0,
            step=0.5
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

        wind_speed = st.number_input(
            "Wind Speed",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.1
        )

        general_diffuse_flows = st.number_input(
            "General Diffuse Flows",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        diffuse_flows = st.number_input(
            "Diffuse Flows",
            min_value=0.0,
            value=80.0,
            step=10.0
        )

    with col2:
        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12
        )

        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=0
        )

        day_of_month = st.slider(
            "Day of Month",
            min_value=1,
            max_value=31,
            value=15
        )

    submitted = st.form_submit_button(
        "Predict Electricity Consumption",
        type="primary",
        use_container_width=True
    )

if submitted:

    # Start with raw feature values
    input_data = {
        "Temperature": temperature,
        "Humidity": humidity,
        "WindSpeed": wind_speed,
        "GeneralDiffuseFlows": general_diffuse_flows,
        "DiffuseFlows": diffuse_flows,
        "Hour": hour,
        "Month": month,
        "DayOfMonth": day_of_month,
        "Weekend": int(day_of_week in ["Saturday", "Sunday"])
    }

    # Add categorical column before encoding
    raw_input = pd.DataFrame([{
        **input_data,
        "DayOfWeek": day_of_week
    }])

    # Match dummy encoding used in training
    encoded_input = pd.get_dummies(
        raw_input,
        columns=["DayOfWeek"],
        drop_first=True,
        dtype=int
    )

    # Add any training columns absent from this input
    for col in feature_columns:
        if col not in encoded_input.columns:
            encoded_input[col] = 0

    # Remove unexpected columns and preserve training order
    encoded_input = encoded_input.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scaled copy for Ridge / Lasso
    scaled_input = scaler.transform(encoded_input)

    # Predict
    try:
        predictions = {}
        for target_name, model in models.items():
            if model_info[target_name]["use_scaler"]:
                predictions[target_name] = model.predict(scaled_input)[0]
            else: 
                predictions[target_name] = model.predict(encoded_input)[0]

        st.divider()
        st.subheader("Prediction Result")
        st.caption("The predicted results are in kilowatts(kW)")

        st.metric(
            label="Predicted Total Electricity Consumption (All Zones)",
            value=f"{predictions['Total (All Zones)']:,.2f} kW"
        )
        
        z1, z2, z3 = st.columns(3)
        z1.metric("Zone 1", f"{predictions['Zone 1']:,.2f} kW")
        z2.metric("Zone 2", f"{predictions['Zone 2']:,.2f} kW")
        z3.metric("Zone 3", f"{predictions['Zone 3']:,.2f} kW")

        st.bar_chart(
            pd.DataFrame({
                "Consumption": [
                    predictions["Zone 1"],
                    predictions["Zone 2"],
                    predictions["Zone 3"]
                ]
            }, index=["Zone 1", "Zone 2", "Zone 3"])
        )

        st.success("Prediction completed successfully!")

        # Display submitted inputs
        with st.expander("View input details"):
            st.dataframe(
                raw_input.T.rename(columns={0: "Value"}),
                use_container_width=True
            )

    except Exception as e:
        st.error(
            "Prediction failed. Check that the app's input "
            "features and encoding match your training data."
        )
        st.exception(e)

# Sidebar
st.sidebar.title("About This Project")
st.sidebar.write(
    """
    **Project:** Electricity Consumption Prediction

    **Dataset:** Power Consumption of Tetouan City
    """
)

for target_name, info in model_info.items():
    st.sidebar.write(
        f"**{target_name}:** {info['model_name']} (R² ≈ {info['r2']:.3f})"
    )
