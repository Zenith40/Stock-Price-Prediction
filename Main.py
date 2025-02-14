import yfinance as yf
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler

st.write("""
         
## Simple Stock Price App
              
""")

st.write('')
st.write('')

df = pd.read_csv('Stock list.csv')

d = {}
for i in range(df.shape[0]):
    d.update({df.Name[i]: df.Symbol[i]})


Stock = st.sidebar.selectbox('Stock', d.keys())
# st.header(Stock)


position = ['Close', 'Open', 'High', 'Low']

pos = st.sidebar.selectbox("Position : ", position)


today = datetime.date(1990, 1, 1)
tomorrow = datetime.date.today()

col1, col2 = st.sidebar.columns(2)

# Place the start date input in the first column
with col1:
    start_date = st.date_input('Start date', today)

# Place the end date input in the second column
with col2:
    end_date = st.date_input('End date', tomorrow)


if start_date < end_date:
    # st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    st.sidebar.success('Fetched Data')
else:
    st.sidebar.error('Error: End date must fall after start date.')

# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75

# define the ticker symbol
tickerSymbol = d[Stock]

# get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# get the historical prices for this ticker

tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
# Open	High	Low	Close	Volume	Dividends	Stock Splits


# checkbox
# check if the checkbox is checked
# title of the checkbox is 'Show/Hide'
# if st.checkbox("Charts"):
    # st.line_chart(tickerDf.Close)
    # st.line_chart(tickerDf.Volume)


st.line_chart(tickerDf[pos])

#------------------------ LOAD DATA -----------------------

stockDf = yf.download(tickerSymbol, start=today, end=tomorrow).reset_index()

# ----------------------- DATAFRAME -----------------------

st.write('')
st.write('')

st.subheader('Data Frame', divider=True)

st.dataframe(stockDf.tail(8))


st.write('')
st.write('')

st.subheader('Data Description', divider=True)


st.write(stockDf.describe())


# ---------------------------- VISUALIZATION -------------------------------------

st.write('')
st.write('')

st.subheader(f'{pos} Price vs Time Chart', divider=True)

fig = plt.figure(figsize=(12, 6))
plt.plot(stockDf.Date, stockDf[pos], 'y')
st.pyplot(fig)


st.write('')
st.write('')

# Visualizing with MA

# 100 & 200 Days Moving Average
ma100 = stockDf[pos].rolling(100).mean()
ma200 = stockDf[pos].rolling(200).mean()

st.subheader(f'{pos} Price vs Time Chart with 100 & 200 MA', divider=True)

fig1 = plt.figure(figsize=(12, 6))
plt.plot(stockDf.Date, stockDf[pos], 'Orange')
plt.plot(stockDf.Date, ma100, 'b')
plt.plot(stockDf.Date, ma200, 'g')
st.pyplot(fig1)

# ---------------------------- SPLIT -------------------------------------

# Splitting Data into Training & Testing

data_training = pd.DataFrame(stockDf[pos][0:int(len(stockDf)*0.70)])
data_testing = pd.DataFrame(stockDf[pos][int(len(stockDf)*0.70):int(len(stockDf))])

# Scaling Data


scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# ---------------------------------- MODEL --------------------------------

model = keras.models.load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)


# Scaling Testing Data

input_data = scaler.transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler_ratio = scaler.scale_
scale_factor = 1/scaler_ratio
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# ----------------- Plot Prediction-------------------------

st.write('')
st.write('')

st.subheader('Prediction Vs Original', divider=True)
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# ----------------- Prediction-------------------------

st.write('')
st.write('')

tomorrow = scaler.transform(final_df.tail(100))
x = model.predict(tomorrow.reshape(1,100,1))


if st.button("Analyze", use_container_width=True):
    st.subheader(f" * {stockDf.iloc[-1,0].strftime('%Y-%m-%d')} : {np.around(np.array(final_df.tail(1))[0][0],2)}")
    st.subheader(f" * {stockDf.iloc[-1,0].strftime('%Y-%m-%d')} prediction Value : {np.around(y_predicted[-1][0],2)}")
    st.subheader(f" * Offset : {np.around((np.array(final_df.tail(1))[0][0])-(y_predicted[-1][0]),2)}")
    st.subheader(f" * Prediction for tomorrow : {np.around(x[0][0]*scale_factor,2)}")
    