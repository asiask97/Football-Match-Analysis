import csv
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('path/to/csv/') 

X = data.iloc[:, 2:-2].values
print(X[1])
y = data.iloc[:, -1].values
print(y[1])
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(y_train)

model = tf.keras.models.Sequential()

#layer one
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#layer two
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#layer three
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(3, activation = tf.nn.sigmoid))

#traning of the model
model.compile(optimizer = 'adam',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=100)

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss, val_acc)

### checking against data collected 
x = 0
correct = 0
wrong = 0
matches = 0
draw = 0
while x < len(X):
    #take the data for two teams
    home = []
    away = []
    home.append(X[x])
    away.append(X[x+1])
   
    #get the difference of win prediction 
    home_predict = model.predict(sc.transform(home))
    away_predict = model.predict(sc.transform(away))
    chance = home_predict[0][2]-away_predict[0][2]
    #comapre to actual result 
    home_win = y[x]
    print(chance, home_win )
    if chance > 0 and home_win == 2:
        correct +=1
    elif chance < 0 and home_win == 1:
        correct +=1
    elif home_win == 0:
        draw +=1
        correct +=1
    else:
        wrong +=1
   
    x += 2
    matches +=1
    
    print(matches, 'out of', len(X)/2)
       
if chance < 0:
    print('home will lose')
if chance is > 0:
    print('home will win')
#win = 2 #lose = 1 #draw = 0