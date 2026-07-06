import streamlit as st

st.set_page_config(
    page_title="Credit Intelligence Engine",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Intelligence Engine")
st.write("Interactive credit risk scoring and decision engine.")

st.info("Streamlit app running successfully.")

st.subheader("Applicant Inputs")

income = st.number_input("Annual income", min_value=0.0, value=150000.0)
credit_amount = st.number_input("Requested credit amount", min_value=0.0, value=500000.0)

if st.button("Evaluate Application"):
    st.success("App structure is working.")
    st.write("Income:", income)
    st.write("Requested credit amount:", credit_amount)
