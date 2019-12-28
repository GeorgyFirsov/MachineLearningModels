# MachineLearningModels

This repository contains machine learning models implementation in Python 3.

You can look for some examples in files Example<model>.ipynb.

### Usage

Usage of these models is really simple - you just need to import required model class and fit it with your data.

#### Generalized linear regression
```python
from MachineLearning.LinearRegression import LinearRegression
from MachineLearning.Metrics import mean_squared_error

model = LinearRegression(degree=3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(mean_squared_error(y_pred, y_test))
```
