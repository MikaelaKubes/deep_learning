# Artificial Neural Network

# Installing Theano
# Installing TensorFlow
# Installing Keras: pip install --upgrade keras

# Part I: Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/MikaelaKubes/Documents/learning_programming/deeplearningA-Z/Artificial_Neural_Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:, 13].values #index 13

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encoding France to 0, Spain to 1, and Germany to 2 in the data
labelencoder_X_1 = LabelEncoder()
# the index 1 corresponds to the independent varibale we want to encode
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
# encoding the gender varible
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])
# our categorical data is not ordinal (no relational order between the categories)
# one is not higher than the other
# only doing it for index 1 since it contains 3 categories
# creating dummy varibles
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #two dummy variables for the countries

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part II: Making the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
# defining it as a sequence of layers 
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu',
                     input_dim=11)) #Dense(output_dim-the number of 
# nodes you want to add in
# this hidden layer, input_dim=the number of nodes in input layer, 
# independent varibles)
# the number of nodes in the input layer is 11 because it's the number of
# independent varibales (X_train) and one node in the output layer
# so the average is 11+1=12/2=6 nodes in the hidden layer

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
# If you are dealng with a dependent varible that has more than two categories
# (ex. 3) then you would need to change two things; the output_dim parameter
# (input 3) and the second thing is activation function to suthmax

# Compiling the ANN
# Applying Stochastic Gradient Descent on the whole artificial neural network
classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                   metrics=['accuracy']) 
# arguments; optimizer-the algorithm you want to use find the optimal number
# of weights; loss-the loss function within the stocastic gradient descent
# algorithm the adam (more than 2 categories=categorical_crossentropy. But
# here we have a binary outcome so use binary_crossentropy


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# Part III: Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# top left square + bottom right square = correct predictions
# bottom left square + top right square = incorrect predictions

# accuracy = number of correct predictions/total number of predictions



# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 
                                                3, 60000, 2, 1, 1, 50000]])))
# second pair of brackets creates 2D array
# to get the value of France from dummy variables compare dataset to X
new_prediction = (new_prediction > 0.5)

# Part IV: Evaluating, Improving, and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu',
                         input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                       metrics=['accuracy']) 
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10,
                             nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                             cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
                         activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
                         activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                         activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_