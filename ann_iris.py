# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('iris_data.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y = labelencoder_y_1.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features= [0])
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()



#Encoding the output classification 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 4))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1, nb_epoch = 150)


y_pred = classifier.predict(X_test)

y_tester = (y_test > 0.5)
y_pred = (y_pred > 0.5)





