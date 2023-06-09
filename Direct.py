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

def create_recursive_data(data, feature, window_size, target_size):
    i = 1
    while(i < window_size):
        data["{}_{}".format(feature, i)] = data[feature].shift(-i)
        i += 1
    i = 0
    while(i < target_size):
        data["{}_{}".format("target", i)] = data[feature].shift(-window_size-i)
        i += 1
    data = data.dropna(axis=0)
    return data

window_size = 5
target_size = 3
data = create_recursive_data(data, "co2", window_size, target_size)
# data = data.drop("time", axis=1)

target = ["target_{}".format(i) for i in range(target_size)]
x = data.drop(["time"] + target, axis=1)
y = data[target]

# Split x train, x test
train_size = 0.8
num_samples = len(x)
num_train = int(train_size * num_samples)

x_train = x[:num_train]
y_train = y[:num_train]
x_test = x[num_train:]
y_test = y[num_train:]

regs = [LinearRegression() for _ in range(target_size)]

for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i)])

r2 = []
mae = []
mse = []

fig, ax = plt.subplots()
ax.plot(data["time"][:num_train], data["co2"][:num_train], label="Train")
ax.plot(data["time"][num_train:], data["co2"][num_train:], label="Test")


for i, reg in enumerate(regs):
    y_pred = reg.predict(x_test)
    ax.plot(data["time"][num_train:], y_pred, label="Predict")
    r2.append(r2_score(y_test["target_{}".format(i)], y_pred))
    mae.append(mean_absolute_error(y_test["target_{}".format(i)], y_pred))
    mse.append(mean_squared_error(y_test["target_{}".format(i)], y_pred))

print("R2 score {}".format(r2))
print("Mean Absolute Error {}".format(mae))
print("Mean Square Error {}".format(mse))

ax.set_xlabel("Year")
ax.set_ylabel("CO2")
ax.grid()
ax.legend()
plt.show()