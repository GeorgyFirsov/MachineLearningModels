# MachineLearningModels

This repository contains machine learning models implementation in Python 3.

You can look for some examples in files Example<model>.ipynb.

### Usage

Usage of these models is really simple:
```python
from MachineLearning.LinearRegression import LinearRegression

model = LinearRegression(degree=3)

model.fit(X_train, y_train)
model.predict(X_test)
```
