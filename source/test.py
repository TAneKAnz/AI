import numpy as np
from sklearn.linear_model import LinearRegression

# Define the sequence and reshape it into a 2D array
sequence = [1, 2, 3, 4, 5, 6]
X = np.array(sequence[:-1]).reshape(-1, 1)
y = np.array(sequence[1:]).reshape(-1, 1)

# Fit the model to the data and make a prediction
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[6]])

print(prediction)  # Output: [[7]]
