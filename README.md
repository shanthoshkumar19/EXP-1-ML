# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries for data handling, visualization, and linear regression.
2. Load the car price dataset and perform basic preprocessing.
3. Split the dataset into training and testing data.
4. Train a linear regression model using the training data.
5. Predict car prices, evaluate the model, and test linear regression assumptions.
 
## Program:
```

 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: SHANTHOSH KUMAR R
RegisterNumber:  212225040402

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('CarPrice_Assignment.csv')

x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

print("Name: SHANTHOSH KUMAR R ")
print("Reg. No: 212225040402 ")
print("MODEL COEFFIECIENTS: ")
for feature , coef in zip(x.columns,model.coef_):
    print(f"{feature:>12}: {coef:10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test,y_pred):>10.2f}")
print(f"{'MAE':>12}: {mean_absolute_error(y_test,y_pred):>10.2f}")
print(f"{'RMSE'}: {np.sqrt(mean_squared_error(y_test,y_pred))}")
print(f"{'R-squared':>12}: {r2_score(y_test,y_pred):>10.2f}")

plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha =0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
    "\n(Values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()

```

## Output:
<img width="301" height="194" alt="image" src="https://github.com/user-attachments/assets/509c349a-10b4-40d0-9710-7a520dadec46" />
<img width="433" height="145" alt="image" src="https://github.com/user-attachments/assets/7ff685ce-210b-4a40-b2be-ade2c0d155ff" />
<img width="849" height="405" alt="image" src="https://github.com/user-attachments/assets/49d42a4b-f38a-47d8-bb4d-b190f2db19b8" />
<img width="525" height="70" alt="image" src="https://github.com/user-attachments/assets/1a1933f0-1cd7-4ae5-b3b9-0a4b9a1afd56" />
<img width="1298" height="582" alt="image" src="https://github.com/user-attachments/assets/842fde6f-6b8e-4427-8bff-1a04c5918fbe" />



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
