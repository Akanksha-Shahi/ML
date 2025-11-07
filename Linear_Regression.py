# Building Linear Regression from Scratch in Python
import numpy as np
# Linear Regression
# initiating the parameters ( learning_rate and no_of_iterations)
class Linear_Regression():
    def __init__(self,learning_rate, no_of_iterations):                 #to initialize ( learning_rate and no_of_iterartions are hyper pararmeteres)   ( weight and bias are model parameters)
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.weight = None
        self.bias = None
    def fit(self,x,y):       #to train the model    
        # no.of training examples and no.of features
        self.m, self.n = x. shape       # no.of rows and columns
        # initaiting the weigt and bias of mour model
        self.w = np.zeros(self.n)   # weight is initialized to zero [ matrix of 0 ]
        self.b = 0                   # bias is initialized to zero
        self.x = x
        self.y = y
        # implementing gadient descent
        for i in range ( self.no_of_iterations):        # for model to iterate over no.of iterations
            self.update_weights()
    def update_weights(self,):                   
        y_pred = self.predict( self.x)      # y prediction
        # calculating gradients
        dw = -(2/self.m) * ( self.x.T.dot( self.y - y_pred))   # gradient of weight
        db = -(2/self.m) * np.sum( self.y - y_pred)            # gradient of bias
        # updating weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
    def predict(self, x):                  # to predict the salary based on years of experience
        return x.dot( self.w)+ self.b      # y = mx + c  ( dot product of x and weight + bias)

# Using Linear Regression model for Predction
# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

# data preprocessing
# loading the data from csv file to a pandas dataframe
salary_data = pd. read_csv('salary_data.csv')
# printing 1st 5 columns of DF
salary_data.head()
# printing last 5 rows of DF
salary_data.tail()
# shape of data
salary_data.shape

# checking for missing values
salary_data.isnull().sum()
# spliting the feature and the target
x = salary_data.iloc[:, :-1].values  # all rows and all columns except last column
y = salary_data.iloc[:, 1].values    # all rows and only last column
print(x)
print(y)

# splitting the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)
# training the model
model = Linear_Regression(learning_rate=0.02, no_of_iterations= 100) #our model    ( the value of model will go in self)
model.fit(x_train, y_train)
# printing the parameter values ( weights and bias )
print("Weight:", model.w [0])
print("Bias:", model.b)

# predict the salary value for the test data
test_data_prediction = model.predict(x_test)
print("Predicted values:", test_data_prediction)
# visualizing the predicted values and actual values
plt.scatter( x_test, y_test, color='blue', label='Actual values')
plt.scatter( x_test, test_data_prediction, color='red', label='Predicted values')
plt.title('Actual vs Predicted values')
plt.plot(x_test, test_data_prediction, color='green')
plt.show()