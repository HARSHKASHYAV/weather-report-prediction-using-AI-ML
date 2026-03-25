import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# load dataset
df = pd.read_csv("seattle-weather.csv")

# preprocessing
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df = df.dropna()

# features and target
X = df[['precipitation', 'temp_min', 'wind', 'month']]
y = df['temp_max']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# models
rf = RandomForestRegressor()
lr = LinearRegression()

# train
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# accuracy
rf_acc = r2_score(y_test, rf.predict(X_test))
lr_acc = r2_score(y_test, lr.predict(X_test))

# save model
pickle.dump((rf, lr, rf_acc, lr_acc), open("weather_model.pkl", "wb"))

print("✅ Models trained successfully")
print("Random Forest Accuracy:", rf_acc)
print("Linear Regression Accuracy:", lr_acc)