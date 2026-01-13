import streamlit as st
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")

st.title("Restaurant Rating Prediction App")

#st.set_page_config(layout = "wide")

st.caption("this app helps you to predic a restaurants review class")

st.divider()

# Average Cost for two	Has Table booking	Has Online delivery	Price range

averagecost = st.number_input("Please enter the estimated average cost for two", min_value = 50, max_value = 9999999, value = 500, step= 50)

tablebooking = st.selectbox("Restaurant has table booking?", ["Yes","No"])

onlinedelivery = st.selectbox("Restaurant has online booking?",["Yes", "No"])

pricerange = st.selectbox("What is the price range (1 -> cheapest, 4 -> Most expensive)", [1,2,3,4])

predictbutton = st.button("predict the review!")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost,bookingstatus ,deliverystatus,pricerange]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)


if predictbutton:
    #st.snow()

    prediction = model.predict(X)

    st.write(prediction)

    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")
    st.divider()