# Logistic Regression
import numpy as np
class Logistic_Regression():
    # initiating the parameters ( learning_rate and no_of_iterations)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.weight = None
        self.bias = None
    # fit function to train the model    
    def fit(self,x,y ):
        self.m, self.n = x.shape       # no.of rows and columns
        self.w = np.zeros(self.n)   # weight is initialized to zero [ matrix of 0 ]
        self.b = 0                   # bias is initialized to zero
        self.x = x
        self.y = y
        for i in range(self.no_of_iterations):        # for model to iterate over no.of iterations
            self.update_weights()
    def update_weights(self):
        # Y_hat (sigmoid function)
        y_hat = 1/(1+ np.exp(-(self.x.dot(self.w)+ self.b)))

        #derivatives
        dw = (1/self.m) * np.dot( self.x.T.dot( y_hat - self.y))   # gradient of weight
        db = (1/self.m) * np.sum( y_hat - self.y)

        # updating the weights and bias using radient descent
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

        # sigmoid equation
    def predict(self, x):
        y_pred = 1/(1+ np.exp(-(x.dot(self.w)+ self.b)))
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred 
# importing dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler            # standardise the data
from sklearn.model_selection import train_test_split   
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# loading the data from csv file to a pandas dataframe
diabetes_data = pd.read_csv('diabetes.csv')
diabetes_data.head()
diabetes_data.shape
diabetes_data.describe()
diabetes_data['Outcome'].value_counts()
diabetes_data.groupby('Outcome').mean()
# separating the data and labels
features = diabetes_data.drop(columns = 'Outcome', axis =1)
target = diabetes_data['Outcome']
print (features)
print (target)

# Data standardization
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)
print(standardized_data)
features = standardized_data
target = diabetes_data['Outcome']
print(features)
print( target)

# train test split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)
print (x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# model training
classifier = Logistic_Regression( learning_rate=0.01, no_of_iterations=100)
classifier.fit(x_train, y_train)                # training the vector support machine classifier
# model evaluation
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy on training data:", training_data_accuracy*100)
# accuracy on test data
x_test_prediction = classifier.predict(x_test)
accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy:", accuracy*100)
# making a predictive system
input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0]== 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")    
# Visualization
plt.scatter(diabetes_data.Age, diabetes_data.Outcome)
plt.xlabel("Age")
plt.ylabel("Outcome")
plt.show()    