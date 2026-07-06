import streamlit as st
import pandas as pd
import joblib
from sklearn import set_config
set_config(transform_output='pandas')

def drop_id(X):
    X = X.copy()
    return X.drop(columns='id')

def add_na_count(X):
    X = X.copy()
    X['na_count'] = X.isna().sum(axis=1)
    return X

def add_prod(X):
    X = X.copy()
    X['activity_x_duration'] = (
        X['ordinal_encoding__physical_activity_level']
        * X['numerical__exercise_duration']
    )
    return X

model = joblib.load('health_model.pkl')

st.title('Health Condition Predictor')

sleep_duration = st.number_input('Sleep Duration', 0.0, 24.0, None)
heart_rate = st.number_input('Heart Rate', 30, 220, None)
bmi = st.number_input('BMI', 10.0, 60.0, None)
calorie_expenditure = st.number_input('Calories Burned', 0, 5000, None)
step_count = st.number_input('Step Count', 0, 50000, None)
exercise_duration = st.number_input('Exercise Duration (minutes)', 0, 300, None)
water_intake = st.number_input('Water Intake (litres)', 0.0, 10.0, None)

diet_type = st.selectbox(
    'Diet',
    ['non-veg', 'balanced', 'veg'],
    index=None
)

stress_level = st.selectbox(
    'Stress Level',
    ['low', 'medium', 'high'],
    index=None

)

sleep_quality = st.selectbox(
    'Sleep Quality',
    ['poor', 'average', 'good'],
    index=None
)

physical_activity_level = st.selectbox(
    'Physical Activity',
    ['sedentary', 'moderate', 'active'],
    index=None
)

smoking_alcohol = st.selectbox(
    'Smoking / Alcohol',
    ['no', 'occasional', 'yes'],
    index=None
)

gender = st.selectbox(
    'Gender',
    ['male', 'female', 'other'],
    index=None
)

if st.button('Predict'):

    sample = pd.DataFrame({
        'id':[0],
        'sleep_duration':[sleep_duration],
        'heart_rate':[heart_rate],
        'bmi':[bmi],
        'calorie_expenditure':[calorie_expenditure],
        'step_count':[step_count],
        'exercise_duration':[exercise_duration],
        'water_intake':[water_intake],
        'diet_type':[diet_type],
        'stress_level':[stress_level],
        'sleep_quality':[sleep_quality],
        'physical_activity_level':[physical_activity_level],
        'smoking_alcohol':[smoking_alcohol],
        'gender':[gender]
    })

    prediction = model.predict(sample)
    st.success(f'Predicted health condition: **{prediction[0]}**')
