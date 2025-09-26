
import pandas as pd
import streamlit as st
import joblib
import sklearn

st.set_page_config(layout= 'wide', page_title='BMW Cars Price Deployment')

# st.title('BMW Cars Prices Prediction')
html_title = """<h1 style="color:white;text-align:center;"> BMW Cars Prices Prediction </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

st.image('https://prod.cosy.bmw.cloud/bmwweb/cosySec?COSY-EU-100-2545xM4RIyFnbm9Mb3AgyyIJrjG0suyJRBODlsrjGpuaprQbhSIqppglBgMxEJl384MlficYiGHqoQxYLW7%25f3tiJ0PCJirQbLDWcQW7%251uNRrqoQh47wMvcYi9t5BJMb3islBglUUJecRScH8R4MbnMdoPeyJGy53LvrQ%25r9YaJW8zWuEJQqogqaFQ7l3ilUjzJcRScH78lMbnMd0zqyJGy5iubrQ%25r9SbUW8zWunDjqogqaG4zl3ilU%25QocRScHzUVMbnMdg4ayJGy5iJUrQ%25r9saYW8zWuKbGqogqaDJKl3ilUCQIcRScH4%25bMbnMdJmSyJGy5Q3SrQ%25r98R5W8zWuobuqogqa3Jnl3ilUR%25gcRScHbU8MbnMdJbkyJGy5Q4ErQ%25r993UW8zWuu3HqogqaaUbl3ilUjv0cRSrQdr9SMBW8zcRacHHwsMbnW85WuEfuqoQEdcNq0zxcqW8JuzM8nq0z6Fboy6oEd82')

df = pd.read_csv('cleaned_df.csv')
st.dataframe(df.head())

model = st.selectbox('Car Model', df.model.unique())
year = st.sidebar.slider('Year', min_value = 1996, max_value = 2020, step= 1)
transmission = st.sidebar.selectbox('Transmission', df.transmission.unique())
mileage = st.text_input('Mileage', '')
fuelType = st.sidebar.selectbox('Fuel type', df.fuelType.unique())
tax = st.text_input('tax', '')
mpg = st.text_input('mpg', '')
engineSize = st.sidebar.selectbox('Engine Size', df.engineSize.unique())

ml_model = joblib.load('catboost.pkl')

if st.button('Predict Car Price'):

    new_data = pd.DataFrame(columns= df.columns.drop('price'), data= [[model, year, transmission, mileage, fuelType, tax, mpg, engineSize]])

    st.write('Car Price :', ml_model.predict(new_data).round(2)[0])
