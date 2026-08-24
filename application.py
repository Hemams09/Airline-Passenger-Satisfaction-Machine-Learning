import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load Model
# ==============================

model = joblib.load("xgboost_airline_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Airline Passenger Satisfaction Prediction",
    page_icon="✈️",
    layout="wide"
)

# ==============================
# Sidebar
# ==============================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Prediction",
        "About Project",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("Best Model : XGBoost")
st.sidebar.info("Accuracy : 96.00%")

# ==========================================
# Prediction Page
# ==========================================

if page == "Prediction":

    st.title("✈ Airline Passenger Satisfaction Prediction")

    st.write(
        "Enter passenger details below to predict whether the passenger is satisfied or not."
    )

    col1, col2 = st.columns(2)

    with col1:

        Gender = st.selectbox(
            "Gender",
            ["Female","Male"]
        )

        Customer_Type = st.selectbox(
            "Customer Type",
            ["Loyal Customer","disloyal Customer"]
        )

        Age = st.number_input(
            "Age",
            7,
            85,
            30
        )

        Type_of_Travel = st.selectbox(
            "Type of Travel",
            ["Business travel","Personal Travel"]
        )

        Class = st.selectbox(
            "Travel Class",
            ["Business","Eco","Eco Plus"]
        )

        Flight_Distance = st.number_input(
            "Flight Distance",
            0,
            5000,
            1000
        )

        Inflight_wifi_service = st.slider(
            "Inflight WiFi Service",
            0,5,3
        )

        Departure_Arrival_time_convenient = st.slider(
            "Departure/Arrival Time Convenient",
            0,5,3
        )

        Ease_of_Online_booking = st.slider(
            "Ease of Online Booking",
            0,5,3
        )

        Gate_location = st.slider(
            "Gate Location",
            0,5,3
        )

        Food_and_drink = st.slider(
            "Food and Drink",
            0,5,3
        )

    with col2:

        Online_boarding = st.slider(
            "Online Boarding",
            0,5,3
        )

        Seat_comfort = st.slider(
            "Seat Comfort",
            0,5,3
        )

        Inflight_entertainment = st.slider(
            "Inflight Entertainment",
            0,5,3
        )

        On_board_service = st.slider(
            "On Board Service",
            0,5,3
        )

        Leg_room_service = st.slider(
            "Leg Room Service",
            0,5,3
        )

        Baggage_handling = st.slider(
            "Baggage Handling",
            0,5,3
        )

        Checkin_service = st.slider(
            "Check-in Service",
            0,5,3
        )

        Inflight_service = st.slider(
            "Inflight Service",
            0,5,3
        )

        Cleanliness = st.slider(
            "Cleanliness",
            0,5,3
        )

        Departure_Delay_in_Minutes = st.number_input(
            "Departure Delay",
            0,
            2000,
            0
        )

        Arrival_Delay_in_Minutes = st.number_input(
            "Arrival Delay",
            0,
            2000,
            0
        )

    # ==============================
    # Encode
    # ==============================

    Gender = 0 if Gender=="Female" else 1

    Customer_Type = 0 if Customer_Type=="Loyal Customer" else 1

    Type_of_Travel = 0 if Type_of_Travel=="Business travel" else 1

    if Class=="Business":
        Class=0
    elif Class=="Eco":
        Class=1
    else:
        Class=2

    # ==============================
    # Predict Button
    # ==============================

    if st.button("Predict Passenger Satisfaction"):

        data = np.array([[
            Gender,
            Customer_Type,
            Age,
            Type_of_Travel,
            Class,
            Flight_Distance,
            Inflight_wifi_service,
            Departure_Arrival_time_convenient,
            Ease_of_Online_booking,
            Gate_location,
            Food_and_drink,
            Online_boarding,
            Seat_comfort,
            Inflight_entertainment,
            On_board_service,
            Leg_room_service,
            Baggage_handling,
            Checkin_service,
            Inflight_service,
            Cleanliness,
            Departure_Delay_in_Minutes,
            Arrival_Delay_in_Minutes
        ]])

        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0]

        st.write("---")

        if prediction == 1:

            st.success("✅ Passenger is Satisfied")

        else:

            st.error("❌ Passenger is Neutral or Dissatisfied")

        st.subheader("Prediction Confidence")

        st.write(f"Probability : {max(probability)*100:.2f}%")
# ==========================================
# About Project
# ==========================================

elif page == "About Project":

    st.title("📖 About Project")

    st.markdown("""
### Project Title

**Airline Passenger Satisfaction Prediction Using Machine Learning**

---

### Project Description

This project predicts whether an airline passenger is **Satisfied** or **Neutral/Dissatisfied**
using Machine Learning algorithms.

The dataset consists of passenger demographic details,
travel information, and airline service ratings.

The objective is to help airlines identify the important
factors influencing passenger satisfaction and improve
overall customer experience.

---

### Algorithms Implemented

✅ Logistic Regression

✅ Decision Tree

✅ Random Forest

✅ K-Nearest Neighbors (KNN)

✅ Naïve Bayes

✅ Support Vector Machine (SVM)

✅ XGBoost

---

### Best Performing Model

🏆 XGBoost

Accuracy : **96.00%**

---

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- XGBoost
- Pandas
- NumPy
- Joblib

---

### Developed By

**Hema M S**

M.Tech – Data Science & Engineering
""")
# ==========================================
# Model Performance
# ==========================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    performance = pd.DataFrame({

        "Model":[

            "Logistic Regression",

            "Decision Tree",

            "Random Forest",

            "KNN",

            "Naïve Bayes",

            "Support Vector Machine",

            "XGBoost"

        ],

        "Accuracy (%)":[

            86.51,

            93.46,

            95.80,

            91.72,

            85.10,

            94.84,

            96.00

        ]

    })

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.write("---")

    st.subheader("Performance Comparison")

    st.bar_chart(
        performance.set_index("Model")
    )

    st.write("---")

    st.subheader("Top Important Features")

    feature = pd.DataFrame({

        "Feature":[

            "Online Boarding",

            "Inflight WiFi",

            "Type of Travel",

            "Travel Class",

            "Inflight Entertainment"

        ],

        "Importance":[

            95,

            90,

            88,

            84,

            82

        ]

    })

    st.dataframe(
        feature,
        use_container_width=True
    )

    st.bar_chart(
        feature.set_index("Feature")
    )

    st.success("Best Performing Model : XGBoost (96%)")