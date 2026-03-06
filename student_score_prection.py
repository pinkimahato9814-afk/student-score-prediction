import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = {
    "hours":[1,2,3,4,5],
    "score":[50,55,65,70,80]
}

df = pd.DataFrame(data)
X = df[['hours']]
y = df['score']
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[6]])
print(prediction)