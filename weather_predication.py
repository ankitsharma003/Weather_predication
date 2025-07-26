

import pandas as pd # for handling and analysing the data
import numpy as np # for numerical operations
from sklearn.model_selection import train_test_split # split data into training and testing sets
from sklearn.preprocessing import LabelEncoder #to convert catogrical data into numerical values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # models for classification and regression tast
from sklearn.metrics import mean_squared_error # to measure accurecy
from datetime import datetime,timedelta # to handle date and time
import pytz
import requests
from dotenv import load_dotenv
import os


API_KEY= os.getenv("API_KEY")
BASE_URL= os.getenv("BASE_URL")

"""1. Fetch current data"""

import requests

def current_weather(city):
    try:
        url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"

        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            print("DEBUG: API Error:", data.get("message"))
            return None

        return {
            'current_temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'Pressure': data['main']['pressure'],
            'humidity': data['main']['humidity'],
            'Visibility': data.get('visibility', 10000),
            'windSpeed': data['wind']['speed'],
            'windDir': data['wind'].get('deg', 0),
            'windGust': data['wind'].get('gust', 0),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
        }

    except Exception as e:
        print("DEBUG: Exception in API call:", str(e))
        return None

"""2. Read historical data"""

def read_historical_data(filename):
  df =pd.read_csv(filename)
  df = df.dropna() # drop with mising data
  df = df.drop_duplicates()
  return df

"""3. prepare data for training"""

def prepare_data(data):
    features = ['mintempC', 'maxtempC', 'FeelsLikeC', 'WindGustKmph',
                'humidity', 'pressure', 'tempC', 'visibility',
                'winddirDegree', 'windspeedKmph']
    X = data[features]
    Y = (data['precipMM'] > 0).astype(int)  # Convert to binary classification label
    return X, Y, None

"""4.Train a model"""

def train_rain_mode(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print("Mean Squared Error for Rain Model:", mean_squared_error(Y_test, Y_pred))
    return model

"""5. prepare regrassion data"""

def prepare_regrassion_data(data, feature):
    X, Y = [], []
    for i in range(len(data)-1):
        X.append(data[feature].iloc[i])
        Y.append(data[feature].iloc[i+1])
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)
    return X, Y

"""6.train regression data"""

def train_regression_model(X, Y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)
    return model

"""7.Predict Future"""

def Predict_Future(model, current_value):
    values = [current_value]
    for _ in range(4):
        next_val = model.predict(np.array(values[-1]).reshape(-1, 1))[0]
        values.append(next_val)
    return values

"""8. **weather analysis**"""

def weather_view():
    city = input("Enter city name: ")
    current_weather_info = current_weather(city)

    if current_weather_info is None:
        print("Weather info could not be retrieved. Please check the city name or API key.")
        return

    historical_data = read_historical_data("/content/jaipur.csv")

    # Prepare data for rain prediction
    X, Y, Le = prepare_data(historical_data)
    rain_model = train_rain_mode(X, Y)

    current_data = {
        'mintempC': current_weather_info['temp_min'],
        'maxtempC': current_weather_info['temp_max'],
        'FeelsLikeC': current_weather_info['feels_like'],
        'WindGustKmph': current_weather_info['windGust'],
        'humidity': current_weather_info['humidity'],
        'pressure': current_weather_info['Pressure'],
        'tempC': current_weather_info['current_temp'],
        'visibility': current_weather_info['Visibility'],
        'winddirDegree': current_weather_info['windDir'],
        'windspeedKmph': current_weather_info['windSpeed'],
    }

    current_df = pd.DataFrame([current_data])
    rain_proba = rain_model.predict_proba(current_df)[0]
    rain_prediction = int(rain_proba[1] >= 0.5)
    confidence = round(rain_proba[1] * 100, 1) if rain_prediction else round(rain_proba[0] * 100, 1)

    # Prepare temperature and humidity prediction
    X_temp, Y_temp = prepare_regrassion_data(historical_data, 'tempC')
    X_hum, Y_hum = prepare_regrassion_data(historical_data, 'humidity')

    temp_model = train_regression_model(X_temp, Y_temp)
    hum_model = train_regression_model(X_hum, Y_hum)

    future_temp = Predict_Future(temp_model, current_data['mintempC'])
    future_hum = Predict_Future(hum_model, current_data['humidity'])

    # Display results
    timeZone = pytz.timezone('Asia/Kolkata')
    now = datetime.now(timeZone)
    next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    future_days = [next_day + timedelta(days=i) for i in range(5)]

    print(f"\nCity: {city}, {current_weather_info['country']}")
    print(f"Current temperature: {current_weather_info['current_temp']}°C")
    print(f"Feels like: {current_weather_info['feels_like']}°C")
    print(f"Minimum temperature: {current_weather_info['temp_min']}°C")
    print(f"Maximum temperature: {current_weather_info['temp_max']}°C")
    print(f"Humidity: {current_weather_info['humidity']}%")
    print(f"Weather description: {current_weather_info['description']}")
    print(f"Rain prediction: {'Yes' if rain_prediction else 'No'} (Confidence: {confidence}%)")

    print("\n📅 Future Temperature Predictions:")
    for day, temp in zip(future_days, future_temp):
        print(f"{day.strftime('%d-%m-%Y')}: {round(temp, 1)}°C")

    print("\n💧 Future Humidity Predictions:")
    for day, hum in zip(future_days, future_hum):
        print(f"{day.strftime('%d-%m-%Y')}: {round(hum, 1)}%")

# Run the program
weather_view()

