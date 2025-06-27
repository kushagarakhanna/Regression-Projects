import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load space-separated data with proper encoding
BosData = pd.read_csv('/Users/kushagarakhanna/Desktop/linearRegression/BostonHousing.csv')
                    
# Features and target
X = BosData.iloc[:, 0:11]
y = BosData.iloc[:, 13]

# Split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=5)

# Train model
reg = LinearRegression()
reg.fit(Xtrain, ytrain)

# Training scores
ytrainpredict = reg.predict(Xtrain)
print('Train MSE =', mean_squared_error(ytrain, ytrainpredict))
print('Train R2 score =', r2_score(ytrain, ytrainpredict), "\n")

# Testing scoress
ytestpredict = reg.predict(Xtest)
print('Test MSE =', mean_squared_error(ytest, ytestpredict))
print('Test R2 score =', r2_score(ytest, ytestpredict))

# Plot
plt.figure()
plt.scatter(ytest, ytestpredict, color='blue', alpha=0.6)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.grid()
plt.show()
