import streamlit as st
import pandas as pd
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Set the page configuration with custom title and icon
st.set_page_config(page_title='Bank Risk Controller Systems', page_icon='', layout='wide')

with open("F:\GUVI\Project\FINAL PROJECT\MODELS\FinalGradientBoostingClassifierr.pkl", "rb") as model:
    gbc = pickle.load(model)

with open("F:\GUVI\Project\FINAL PROJECT\MODELS\label_encoders.pkl", "rb") as lemodel:
    le = pickle.load(lemodel)

data = pd.read_csv("F:\GUVI\Project\FINAL PROJECT\Model Data.csv", index_col=0)
eda_data = pd.read_csv("F:\GUVI\Project\FINAL PROJECT\Final_cleaned_data.csv")

# Initialize a hidden variable for the active section
if 'active_section' not in st.session_state:
    st.session_state.active_section = "Home"  # Default to "Home"

# Sidebar buttons for navigation
with st.sidebar:
    # st.header("Syed Abuthahir")
    st.title("Select options")
    if st.button("Home"):
        st.session_state.active_section = "Home"
    if st.button("Data"):
        st.session_state.active_section = "Data"
    if st.button("EDA Visual"):
        st.session_state.active_section = "EDA Visual"
    if st.button("Prediction"):
        st.session_state.active_section = "Prediction"
    if st.button("Chatbot"):
        st.session_state.active_section = "Chatbot"

# Define individual sections
def home():
    st.title("Welcome to the Bank Risk Controller Systems!")
    st.subheader("Select a section from the sidebar to get started.")
    st.image(image="F:\GUVI\Project\FINAL PROJECT\Risk.jpeg", width=750)

def models_data():
    st.header("Model Data")
    df = data.head(5000)
    st.dataframe(df)

    st.header("Model Performance")
    performace = pd.DataFrame({
        "Algorithm" : ["GradientBoostingClassifier", "XGBClassifier"],
        "Accuracy_score" : [0.97, 0.96],
        "Precision_score" : [0.97, 0.96],
        "Recall_score" : [0.97, 0.96],
        "F1_score" : [0.97, 0.96]
    })
    st.dataframe(performace)

    st.header("Selected Model")
    st.subheader("*GradientBoostingClassifier*")


def EDA():
    st.title("Insights")

    df = eda_data.head(5)
    st.dataframe(df)

    st.subheader("Describe")
    des = eda_data.describe()
    st.table(des)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Variable Distribution")
        st.caption("Understand the balance of the target variable (0 vs 1)")
        fig, ax = plt.subplots()
        sns.countplot(x='TARGET', data=eda_data, ax=ax, palette='viridis')
        ax.set_title('Target Variable Distribution')
        ax.set_xlabel('Target (0 = No Payment Difficulties, 1 = Payment Difficulties)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        st.subheader("Contract Type vs Average Credit Amount")
        st.caption("Compare average credit amounts for different NAME_CONTRACT_TYPE_x")
        fig, ax = plt.subplots()
        avg_credit = eda_data.groupby('NAME_CONTRACT_TYPE_x')['AMT_CREDIT_x'].mean().reset_index()
        sns.barplot(x='NAME_CONTRACT_TYPE_x', y='AMT_CREDIT_x', data=avg_credit, palette='coolwarm', ax=ax)
        ax.set_title('Average Credit Amount by Contract Type')
        ax.set_xlabel('Contract Type')
        ax.set_ylabel('Average Credit Amount')
        st.pyplot(fig)

        st.subheader("Distribution of Income Types")
        st.caption("Analyze the distribution of NAME_INCOME_TYPE")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(y='NAME_INCOME_TYPE', data=eda_data, order=eda_data['NAME_INCOME_TYPE'].value_counts().index, palette='mako', ax=ax)
        ax.set_title('Distribution of Income Types')
        ax.set_xlabel('Count')
        ax.set_ylabel('Income Type')
        st.pyplot(fig)

    with col2:
        st.subheader("Income vs Credit Amount")
        st.caption("Explore the relationship between AMT_INCOME_TOTAL and AMT_CREDIT_x")
        fig, ax = plt.subplots()
        sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT_x', hue='TARGET', data=eda_data, alpha=0.7, ax=ax)
        ax.set_title('Income vs Credit Amount (Colored by Target)')
        ax.set_xlabel('Total Income')
        ax.set_ylabel('Credit Amount')
        st.pyplot(fig)

        st.subheader("Age Distribution")
        st.caption("Visualize the age distribution (DAYS_BIRTH_YEARS) in the dataset")
        fig, ax = plt.subplots()
        sns.histplot(eda_data['DAYS_BIRTH_YEARS'], bins=20, kde=True, color='blue', ax=ax)
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age (Years)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        st.subheader("Occupation Type Analysis")
        st.caption("Distribution of clients by OCCUPATION_TYPE")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(y='OCCUPATION_TYPE', data=eda_data, order=eda_data['OCCUPATION_TYPE'].value_counts().index, palette='crest', ax=ax)
        ax.set_title('Distribution of Occupation Types')
        ax.set_xlabel('Count')
        ax.set_ylabel('Occupation Type')
        st.pyplot(fig)





def prediction():
    st.title("Prediction Interface")
    col1, col2, col3 = st.columns(3)

    with col1:
        NFLAG_INSURED_ON_APPROVAL = st.selectbox("Insurance Requested During Previous Application",
                                                data['NFLAG_INSURED_ON_APPROVAL'].unique())
        AMT_REQ_CREDIT_BUREAU_YEAR = st.number_input("Credit Bureau Enquiries in Last Year", min_value=0.0, max_value=25.0)
        AMT_CREDIT_x = st.number_input("Loan Credit Amount")
        DAYS_BIRTH_YEARS = st.number_input("Client's Age")
        AMT_ANNUITY_y = st.number_input("Previous Application Annuity")
    with col2:
        DAYS_LAST_PHONE_CHANGE_YEARS = st.number_input("Years Since Last Phone Change", min_value=0.0, max_value=12.00)
        AMT_ANNUITY_x = st.number_input("Loan Annuity")
        NAME_CONTRACT_STATUS = st.selectbox("Previous Application Contract Status", data["NAME_CONTRACT_STATUS"].unique())
        AMT_INCOME_TOTAL = st.number_input("Client's Total Income")
        DAYS_ID_PUBLISH_YEARS = st.number_input("Years Since Last ID Update", min_value=0.0, max_value=20.00)
    with col3:
        OBS_60_CNT_SOCIAL_CIRCLE = st.number_input("Social Surroundings with 60 Days Past Due Defaults")
        NAME_CONTRACT_TYPE_x = st.selectbox("Loan Type (Cash or Revolving)", data["NAME_CONTRACT_TYPE_x"].unique())
        OCCUPATION_TYPE = st.selectbox("Client's Occupation Type", data["OCCUPATION_TYPE"].unique())

    numerical_data = pd.DataFrame({
        "NFLAG_INSURED_ON_APPROVAL" : [NFLAG_INSURED_ON_APPROVAL],
        "AMT_REQ_CREDIT_BUREAU_YEAR" : [AMT_REQ_CREDIT_BUREAU_YEAR],
        "AMT_CREDIT_x" : [AMT_CREDIT_x],
        "DAYS_BIRTH_YEARS" : [DAYS_BIRTH_YEARS],
        "AMT_ANNUITY_y" : [AMT_ANNUITY_y],
        "DAYS_LAST_PHONE_CHANGE_YEARS" : [DAYS_LAST_PHONE_CHANGE_YEARS],
        "AMT_ANNUITY_x" : [AMT_ANNUITY_x],
        'NAME_CONTRACT_STATUS': [NAME_CONTRACT_STATUS],
        "AMT_INCOME_TOTAL" : [AMT_INCOME_TOTAL],
        "DAYS_ID_PUBLISH_YEARS" : [DAYS_ID_PUBLISH_YEARS],
        "OBS_60_CNT_SOCIAL_CIRCLE" : [OBS_60_CNT_SOCIAL_CIRCLE],
        'NAME_CONTRACT_TYPE_x': [NAME_CONTRACT_TYPE_x],
        'OCCUPATION_TYPE': [OCCUPATION_TYPE]
    })


    # Transform categorical features using LabelEncoders
    for column in ['NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE_x', 'OCCUPATION_TYPE']:
        numerical_data[column] = le[column].transform(numerical_data[column])



    if st.button("Predict"):
        # Perform the prediction
        predicted_scaled = gbc.predict(numerical_data)

        if predicted_scaled[0] == 0:
            st.success("No Payment Difficulties.")
        else:
            st.error("Payment Difficulties.")


# if st.sidebar.button("Load Prediction Interface"):
#     prediction()


def chatbot():
    st.title("Chatbot")
    st.write("Chatbot implementation goes here.")

# with st.sidebar:
#     st.button(label="Chatbot", on_click=chatbot)


# Display the active section
if st.session_state.active_section == "Home":
    home()
elif st.session_state.active_section == "Data":
    models_data()
elif st.session_state.active_section == "EDA Visual":
    EDA()
elif st.session_state.active_section == "Prediction":
    prediction()
elif st.session_state.active_section == "Chatbot":
    chatbot()