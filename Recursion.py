import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv("dataset\co2.csv")
# print(data.info())

# Convert feature time to date time type
data["time"] = pd.to_datetime(data["time"])
# print(data["time"].dtype)

# Visualization
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Year")
# ax.set_ylabel("CO2")
# plt.savefig("Plot_Before_Interpolate")

# print(data.isna().sum())

data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Year")
# ax.set_ylabel("CO2")
# plt.savefig("Plot_After_Interpolate")

# data["week_1"] = data["co2"].shift(-1)
# data["week_2"] = data["co2"].shift(-2)
# data["week_3"] = data["co2"].shift(-3)
# data["week_4"] = data["co2"].shift(-4)
# data["target"] = data["co2"].shift(-5)

def create_recursive_data(data, feature, window_side):
    i = 1
    while (i < window_side):
        data["{}_{}".format(feature, i)] = data[feature].shift(-i)
        i += 1
    data["target"] = data[feature].shift(-i)
    data = data.dropna(axis=0)
    return data

data = create_recursive_data(data, "co2", window_side=5)
# data = data.drop("time", axis=1)

target = "target"
x = data.drop([target, "time"], axis=1)
y = data[target]

# Split x train, x test
train_size = 0.8
num_samples = len(x)
num_train = int(train_size * num_samples)

x_train = x[:num_train]
y_train = y[:num_train]
x_test = x[num_train:]
y_test = y[num_train:]

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print("R2 Score: {}".format(r2_score(y_test, y_pred)))
print("MAE Score: {}".format(mean_absolute_error(y_test, y_pred)))
print("MSE Score: {}".format(mean_squared_error(y_test, y_pred)))

fig, ax = plt.subplots()
ax.plot(data["time"][:num_train], data["co2"][:num_train], label="Train")
ax.plot(data["time"][num_train:], data["co2"][num_train:], label="Test", linewidth=3)
ax.plot(data["time"][num_train:], y_pred, label="Predict")
ax.set_xlabel("Year")
ax.set_ylabel("CO2")
ax.grid()
ax.legend()
plt.savefig("Result_Recursion")