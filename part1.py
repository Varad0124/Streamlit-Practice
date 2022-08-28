import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

# add custom CSS
st.markdown(
    """
    <style>
    .main{
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)



@st.cache
def get_data(filename):
    Taxi_Data = pd.read_csv(filename)

    return Taxi_Data

with header:
    st.title('Welcome to My DataScience Project')
    st.text('This Project explains in brief about NYC Taxi Data')


with dataset:
    st.header('New York City Taxi Dataset')

    Taxi_Data = get_data('taxi_tripdata.csv')
    st.write(Taxi_Data.head(5))

    st.subheader('Pick-up Loaction ID distribution on te NYC dataset')
    pickup_location_dist = pd.DataFrame(Taxi_Data['PULocationID'].value_counts())
    st.bar_chart(pickup_location_dist.head(50))

with features:
    st.header('The Features Created')
    st.markdown('* **first feature:** This is Feature 1')
    st.markdown('* **second feature:** Second Feature 2')

with modelTraining:
    st.header('Time to Train the Model') 

    sel_col, disp_col =st.columns(2) 

    max_depth = sel_col.slider('What Should be the depth of the model?', 
    min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('how many trees should be there?', options=[100,200,300,'No limit'], index=0)
    sel_col.write(Taxi_Data.columns)


    input_feature = sel_col.text_input('Which feature should be used as the input feature?','PULocationID')

    if n_estimators=='No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    x = Taxi_Data[[input_feature]]
    y = Taxi_Data[['trip_distance']]

    regr.fit(x,y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean Absolute Error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean Squared Error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared of the model is:')
    disp_col.write(r2_score(y,prediction))
