# Mini Tutorial - Timeseries prediction with LSTM
#
# For questions please contact: support@copadata.com
#
# The tutorial is based on the zenon project PREDICTIVE_MAINTENANCE_DEMO_820.
# In that project a simple simulation creates cyclic welding data.
# To follow this tutorial you will need:
#   - a zenon supervisor > V 8.20 to run the project
#   - a zenon analyzer > V 3.40  set up with a filled meta db for the project
#   - pyZAN installed (pip install CopaData)
#
# Prediction (or better: forecasting) a timeseries is not an easy task. There
# are a lot of different algorithms you can use, that rely on 
# statistics/analytics. Examples would be SARIMA, Exponential Smoothing or
# any kind of linear regression. These algorithms all have their strengths and
# weaknesses and are suited better for certain kinds of timeseries data.
# In an "usual" data science workflow, you would have a look at your data and
# its metrics and select an algorithm based on things like stationarity.
# So sadly there is no single algorithm, that can do it all... but maybe one,
# that can do a lot. Although in most cases using LSTM
# (Long-Short-Term-Memory) neural networks for timeseries forecasting is a little
# overkill, it will in my experience deliver mostly good results.
#
# In this tutorial we will load data from the PREDICTIVE_MAINTENANCE_DEMO_820 
# zenon project and train a LSTM network to forecast data
# for the next welding cycle(~30 seconds).

from CopaData import pyZAN
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

import pandas as pd




# ------------------------------------------------------------
# ---------------- Part 1 - Read trainig data ----------------
# ------------------------------------------------------------

# First connect to our analyzer
zan = pyZAN.Server(server='localhost', metadb='ZA_Predictive820')

# Get projects, archives and variables
projects = zan.read_MetaData_Projects()
archives = zan.read_MetaData_Archives()
variables = zan.read_MetaData_Variables()

                                                        
# We will focus on R1_WeldingCurrent and use 1 hour of data for training

train_data = zan.read_Online_Archive(project_reference = "PREDICTIVE_MAINTENANCE_DEMO_820",\
                                 archive_reference = "PA",\
                                 variable_references = ["RobotSimForPA/Global/R1_WeldingCurrent"],\
                                 time_from = datetime.datetime(2019,12,3,7,20,0),\
                                 time_to = datetime.datetime(2019,12,3,7,50,0),\
                                 show_visualnames=True)



# Our zenon data has a lot of really usefull columns like STATUSFLAGS and UNIT
# For our simple purposes we don't need them... we'll discard all columns but VALUE
train_data = train_data[['VALUE']].VALUE.values.reshape(-1,1)
train_data = np.array(train_data)

# I will save my train dataset.. you could just load it and won't need the zenon
# environment
#train_data = np.load("train_data.npy")

# ------------------------------------------------------------
# ----------------- Part 2 - Reshape the data ----------------
# ------------------------------------------------------------

# When working with any kind of neural network it is usually a good idea to
# to normalize your data to a fixed interval. In this case we will scale it to
# the interval [0,1]
scaler = MinMaxScaler(feature_range = (0, 1))

train_data_scaled = scaler.fit_transform(train_data)

# We want to predict at least 35 seconds of our timeseries. To do this there are
# two different approaches:
#   1) Use n values from the past to predict the value for T+1.
#      Repeat that process using your own predictions as input values to predict
#      T+2, T+3, T+4 ...
#   2) Use n values from the past to predict values for the sequence [T+1,T+35]
# As you can imagine sequence prediction is a lot more challenging for a
# ml model. While using your own predictions as inputs will bring the disadvantage
# of stackig errors.
# We will train a model for each approach and see which one is better.


# When training an neuronal network in most cases you will have to provide two
# arrays of data. One will be the "features", which is the input-data for your
# model and has the same form as the data you will provide to the model later on
# when you want it to calculate a prediction. In our case this will be the values
# from the last 35 timesteps. (35 because we know, that one cylce has roughly
# 32 seconds, so 35 seconds hold at least one complete cycle and should be
# engough for our model to derive the current position in that cycle)
# Corresponding to these features you will have to provide the "labels", or the 
# "ground truth" to the model. During training your model we (hopefully) learn
# how the features influence the labels and will then be able to calculate correct
# predictions for a new set of features presented to it.
# In our case the labels will be the value for T+1 or the values for [T+1,T+35]
# depending on our approach.

# We will now build those datasets. For each second in our dataset we will create
# one row in our features and labels.
# The features will be the same for both models, labels1 will hold the labels 
# for our first model, labels35 for the second

features=[]
labels1=[]
labels35=[]

for i in range(34,train_data_scaled.size-35):
    features.append(train_data_scaled[i-34:i+1,0])
    labels35.append(train_data_scaled[i+1:i+36,0])
    labels1.append(train_data_scaled[i+1,0])

# convert both to np arrays
features = np.array(features)
labels35 =  np.array(labels35)
labels1 =  np.array(labels1)

# ------------------------------------------------------------
# ----------------- Part 3 - the LSTM models ------------------
# ------------------------------------------------------------

# each LSTM network will need a 3 dimensional array as input, with dimensions:
# 1 = nr of rows in the dataset
# 2 = nr of columns / timesteps we're looking back
# 3 = nr of features... only 1 in our case

features = np.reshape(features,(features.shape[0],features.shape[1],1))

# I will save my features and labels, so you can use them without the need for
# the zenon environment

#features = np.load("features.npy")
#labels35 = np.load("labels35.npy")
#labels1 = np.load("labels1.npy")

# create our first model
model1 = Sequential()

# we will combine 3 LSTM layers with one dropout layer each. The dropout layer
# will prevent overfitting to a certain extent, by ignoring a percentage of the
# neurons of the preceding layer.
model1.add(LSTM(units=35, return_sequences=True, input_shape=(features.shape[1], 1)))
model1.add(Dropout(0.2))

model1.add(LSTM(units=35, return_sequences=True))
model1.add(Dropout(0.2))

# --- optional layer---
model1.add(LSTM(units=35, return_sequences=True))
model1.add(Dropout(0.2))
# ---optional layer---

model1.add(LSTM(units=35))
model1.add(Dropout(0.2))

# a dense layer for output
model1.add(Dense(units = 1))

# compile model
model1.compile(optimizer='adam', loss = 'mse')

# finally it's time to train our first model
# depending on your hardware and the setup of your model this can take a while
# The 3 layer version should take about 30 min on an i7, 4 layers will need up
# to 4 hours
# Training will consume a lot (all) of your memory.
# Sometimes memory isn't freed correctly after training, this will make the next
# training crash the python kernel on most computers
# Restarting the kernel, reloading our dataset and proceeding with the second
# training works usually
# I saved my trained models, so you can load them...
model1.fit(features,labels1,epochs=200)

# since this took a while lets save our trained model to disk
model1.save('LSTM Model 1_4Layers.h5')

# you can load your model later with
#model1 = load_model('LSTM Model 1_4Layers.h5')

# create our second model
model2 = Sequential()

model2.add(LSTM(units=35, return_sequences=True, input_shape=(features.shape[1], 1)))
model2.add(Dropout(0.2))

model2.add(LSTM(units=35, return_sequences=True))
model2.add(Dropout(0.2))

# --- optional layer---
model2.add(LSTM(units=35, return_sequences=True))
model2.add(Dropout(0.2))
# --- optional layer---

model2.add(LSTM(units=35))
model2.add(Dropout(0.2))

model2.add(Dense(units = 35))

# compile model
model2.compile(optimizer='adam', loss = 'mse')

model2.fit(features,labels35,epochs=200)

# since this took a while lets save our trained model to disk
model2.save('LSTM Model 2_4Layers.h5')

# you can load your model later with
#model2 = load_model('LSTM Model 2_4Layers.h5')


# ------------------------------------------------------------
# ---------------- Part 3 - make a forecast ------------------
# ------------------------------------------------------------

# Now we wil compare the two models by prediction the same timeframe
# and comparing the results
# let's load a new dataset from zenon
eval_data = zan.read_Online_Archive(project_reference = "PREDICTIVE_MAINTENANCE_DEMO_820",\
                                 archive_reference = "PA",\
                                 variable_references = ["RobotSimForPA/Global/R1_WeldingCurrent"],\
                                 time_from = datetime.datetime(2019,12,3,7,20,0),\
                                 time_to = datetime.datetime(2019,12,3,7,25,0),\
                                 show_visualnames=True)


eval_data=eval_data.VALUE.values.reshape(1,-1)

# I will save my eval data, so you can use them without the need for
# the zenon environment
#eval_data = np.load("eval_data.npy")

eval_data_scaled = scaler.transform(eval_data)



# For our first model we need to make 35 predictions one after another
# After each prediction we will add our last prediction to the input values
d=eval_data_scaled[:,0:35]
prediction1=np.array([[]])
for i in range (0,35):
    p=model1.predict(np.reshape(d,(1,35,1)))
    prediction1 = np.hstack((prediction1,p))
    d=d[:,1:35]
    d= np.hstack((d,p))
    
d = eval_data_scaled[:,0:35]
prediction2 = model2.predict(np.reshape(d,(1,35,1)))

# scale it back
prediction1 = scaler.inverse_transform(prediction1.reshape(-1,1))
prediction2 = scaler.inverse_transform(prediction2.reshape(-1,1))

# plot the results
xv=range(0,80)
plt.subplots()

plt.plot(xv[0:35],eval_data.reshape(-1,1)[0:35], color='b')
plt.plot(xv[35:70],prediction1.reshape(-1,1)[0:35], color='g',label='prediction1')
plt.plot(xv[35:70],prediction2.reshape(-1,1)[0:35], color='r',label='prediction35')
plt.plot(xv[35:80],eval_data.reshape(-1,1)[35:80], color='b',label='real',linestyle='dashed')
plt.legend()

# Both predictions come out quite good, but our model 1 definitely has a better
# accuracy.
