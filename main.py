import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Function to predict likelihood based on user input
def predict_likelihood(dr_recc_h1n1_vacc, is_h1n1_risky, is_h1n1_vacc_effective,
                       age_bracket, is_seas_risky, sick_from_h1n1_vacc, h1n1_worry,
                       sick_from_seas_vacc, qualification, census_msa,
                       is_seas_vacc_effective, no_of_adults, no_of_children,
                       h1n1_awareness, income_level):
    features = np.array([[dr_recc_h1n1_vacc, is_h1n1_risky, is_h1n1_vacc_effective,
                          age_bracket, is_seas_risky, sick_from_h1n1_vacc, h1n1_worry,
                          sick_from_seas_vacc, qualification, census_msa,
                          is_seas_vacc_effective, no_of_adults, no_of_children,
                          h1n1_awareness, income_level]])
    prediction = model.predict(features)
    return prediction[0]

def main():
    # Sidebar menu options
    menu = st.sidebar.selectbox('Menu', ['Model Prediction','Dashboard'])

    if menu == 'Model Prediction':
        display_prediction()
    elif menu == 'Dashboard':
        display_dashboard()

def display_prediction():
    st.title('H1N1 Vaccine Prediction')

    # Collect user input for features
    dr_recc_h1n1_vacc = st.number_input('Doctor recommendation for H1N1 vaccine (0 or 1)', min_value=0, max_value=1)
    is_h1n1_risky = st.slider('Perceived risk of H1N1 flu (0 to 4)', min_value=0, max_value=4, step=1)
    is_h1n1_vacc_effective = st.slider('Perceived effectiveness of H1N1 vaccine (0 to 4)', min_value=0, max_value=4, step=1)
    age_bracket = st.slider('Age bracket (0 to 3)', min_value=0, max_value=3, step=1)
    is_seas_risky = st.slider('Perceived risk of seasonal flu (0 to 4)', min_value=0, max_value=4, step=1)
    sick_from_h1n1_vacc = st.slider('Concerns about getting sick from H1N1 vaccine (0 to 4)', min_value=0, max_value=4, step=1)
    h1n1_worry = st.slider('Worry about H1N1 flu (0 to 3)', min_value=0, max_value=3, step=1)
    sick_from_seas_vacc = st.slider('Concerns about getting sick from seasonal flu vaccine (0 to 4)', min_value=0, max_value=4, step=1)
    qualification = st.slider('Education level (0 to 3)', min_value=0, max_value=3, step=1)
    census_msa = st.number_input('Metropolitan statistical area (MSA) status (0 or 1)', min_value=0, max_value=1)
    is_seas_vacc_effective = st.slider('Perceived effectiveness of seasonal flu vaccine (0 to 4)', min_value=0, max_value=4, step=1)
    no_of_adults = st.number_input('Number of adults in household', min_value=0)
    no_of_children = st.number_input('Number of children in household', min_value=0)
    h1n1_awareness = st.slider('H1N1 flu awareness (0 to 2)', min_value=0, max_value=2, step=1)
    income_level = st.slider('Income level (0 to 3)', min_value=0, max_value=3, step=1)

    # Predict button
    if st.button('Predict'):
        prediction = predict_likelihood(dr_recc_h1n1_vacc, is_h1n1_risky, is_h1n1_vacc_effective,
                                        age_bracket, is_seas_risky, sick_from_h1n1_vacc, h1n1_worry,
                                        sick_from_seas_vacc, qualification, census_msa,
                                        is_seas_vacc_effective, no_of_adults, no_of_children,
                                        h1n1_awareness, income_level)
        st.write(f'Likelihood of taking H1N1 vaccine: {prediction}')



def display_dashboard():
    # Load dataset
    df = pd.read_csv('dataset.csv')

    # Sidebar for filters
    st.sidebar.header('Filter Options')
    sex_options = st.sidebar.multiselect('Sex:', df['sex'].unique(), default=df['sex'].unique())
    age_bracket_options = st.sidebar.multiselect('Age Bracket:', df['age_bracket'].unique(), default=df['age_bracket'].unique())
    income_level_options = st.sidebar.multiselect('Income Level:', df['income_level'].unique(), default=df['income_level'].unique())
    h1n1_worry_options = st.sidebar.multiselect('H1N1 worry:', df['h1n1_worry'].unique(), default=df['h1n1_worry'].unique())

    # Filter dataset based on selections
    filtered_df = df[(df['sex'].isin(sex_options)) &
                     (df['age_bracket'].isin(age_bracket_options)) &
                     (df['income_level'].isin(income_level_options)) &
                     (df['h1n1_worry'].isin(h1n1_worry_options))]

    # Dashboard title
    st.title('H1N1 Vaccine Data Dashboard')

    # Dynamic charts
    st.header('Dynamic Charts')

    # Function to create and display matplotlib charts
    def create_chart(fig, title):
        st.subheader(title)
        st.pyplot(fig)

    # Distribution of H1N1 Vaccine
    fig1, ax1 = plt.subplots()
    sns.countplot(x='h1n1_vaccine', data=filtered_df, ax=ax1)
    ax1.set_title('Distribution of H1N1 Vaccine')
    create_chart(fig1, 'Distribution of H1N1 Vaccine')

    # Categorical Features vs. Target
    fig2, ax2 = plt.subplots()
    sns.countplot(x='sex', hue='h1n1_vaccine', data=filtered_df, ax=ax2)
    ax2.set_title('Sex vs H1N1 Vaccine')
    create_chart(fig2, 'Sex vs H1N1 Vaccine')

    fig3, ax3 = plt.subplots()
    sns.countplot(x='age_bracket', hue='h1n1_vaccine', data=filtered_df, ax=ax3)
    ax3.set_title('Age Bracket vs H1N1 Vaccine')
    create_chart(fig3, 'Age Bracket vs H1N1 Vaccine')

    fig4, ax4 = plt.subplots()
    sns.countplot(x='income_level', hue='h1n1_vaccine', data=filtered_df, ax=ax4)
    ax4.set_title('Income Level vs H1N1 Vaccine')
    create_chart(fig4, 'Income Level vs H1N1 Vaccine')

    fig5, ax5 = plt.subplots()
    sns.countplot(x='h1n1_worry', hue='h1n1_vaccine', data=filtered_df, ax=ax5)
    ax5.set_title('H1N1 worry vs H1N1 Vaccine')
    create_chart(fig5, 'H1N1 worry vs H1N1 Vaccine')


if __name__ == '__main__':
    main()