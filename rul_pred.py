# Imports

import numpy as np
import pandas as pd
import utils
import pickle
from pprint import pprint
from random import sample
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from hmmlearn.hmm import GMMHMM

# Loading data and adding RUL labels

def process_db():
    
    labels = []
    for engine_no_df in gb:
        instances = engine_no_df[1].shape[0]
        label = [instances - i - 1 for i in range(instances)]
        labels += label
    data['RUL'] = labels
    print(data)
    
# Loading model

def get_model(path):
    
    print("Loading model")
    # model = load_model(path)
    model = tf.keras.models.load_model(path)
    return model

# Saving model

def save_model(model, path):
    
    print("Saving model")
    model.save(path)    

# Training model

def train_model():
    
    X1 = data.iloc[:,2:5]
    y1 = data.iloc[:,5:26]
    sc1 = StandardScaler()
    X1 = sc1.fit_transform(X1)
    print(X1)
    y1 = np.asarray(y1)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.1)
    model = Sequential()
    model.add(Dense(21, input_dim = 3, activation = 'relu'))
    model.add(Dense(21, activation = 'relu'))
    model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
    history1 = model.fit(X_train1, y_train1, validation_data = (X_test1, y_test1), epochs = 50)
    y_pred1 = model.predict(X_test1)
    a1 = 0
    for i in range(y_pred1.shape[0]):
        for j in range(y_pred1.shape[1]):
            a1 += abs(y_test1[i][j] - y_pred1[i][j])/y_test1[i][j]
    a1 /= y_pred1.shape[0]*y_pred1.shape[1]
    print('Error: ', a1)
    return model
    
# Performing K-Means Clustering

def perform_k_means():
    
    X_and_y = data.iloc[:, 2:26]

    kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300)
    kmeans.fit(X_and_y)
    
    return X_and_y, kmeans

# Reducing no. of samples

def reduce_samples(X_and_y, kmeans):
    
    X_and_y_1 = []
    X_and_y_2 = []
    X_and_y_3 = []
    X_and_y_4 = []
    X_and_y_w_engines = data.iloc[:, 2:26]
    X_and_y_w_engines.insert(0, 'engine_no', data.iloc[:, 0])
    group = X_and_y_w_engines.groupby(['engine_no'])
    for i in range(X_and_y.shape[0]):
        if kmeans.labels_[i] == 0:
            X_and_y_1.append(X_and_y_w_engines.iloc[i, :])
        elif kmeans.labels_[i] == 1:
            X_and_y_2.append(X_and_y_w_engines.iloc[i, :])
        elif kmeans.labels_[i] == 2:
            X_and_y_3.append(X_and_y_w_engines.iloc[i, :])
        else:
            X_and_y_4.append(X_and_y_w_engines.iloc[i, :])
    X_and_y_1 = sample(X_and_y_1, 1000)
    X_and_y_2 = sample(X_and_y_2, 1000)
    X_and_y_3 = sample(X_and_y_3, 1000)
    X_and_y_4 = sample(X_and_y_4, 1000)
    X_and_y_samples = []
    X_and_y_samples.extend(X_and_y_1)
    X_and_y_samples.extend(X_and_y_2)
    X_and_y_samples.extend(X_and_y_3)
    X_and_y_samples.extend(X_and_y_4)
    
    return X_and_y_samples


# Training model on reduced samples

def train_on_reduced(X_and_y_samples, model):

    X2 = []
    y2 = []
    for l in X_and_y_samples:
        X2.append(l[1:4])
        y2.append(l[4:])
    sc2 = StandardScaler()
    X2 = sc2.fit_transform(X2)
    y2 = np.asarray(y2)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.1)
    history2 = model.fit(X_train2, y_train2, validation_data = (X_test2, y_test2), epochs = 200)
    y_pred2 = model.predict(X_test2)
    a2 = 0
    for i in range(y_pred2.shape[0]):
        for j in range(y_pred2.shape[1]):
            a2 += abs(y_test2[i][j] - y_pred2[i][j])/y_test2[i][j]
    a2 /= y_pred2.shape[0]*y_pred2.shape[1]
    print('Error: ', a2)
    return model

# Feature extraction

def feature_extract(reduced_model):

    flights = pd.DataFrame(X_and_y_samples , columns =['engine_no', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3', 'sensor_measurement_1', 'sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4', 'sensor_measurement_5', 'sensor_measurement_6', 'sensor_measurement_7', 'sensor_measurement_8', 'sensor_measurement_9', 'sensor_measurement_10', 'sensor_measurement_11', 'sensor_measurement_12', 'sensor_measurement_13', 'sensor_measurement_14', 'sensor_measurement_15', 'sensor_measurement_16', 'sensor_measurement_17', 'sensor_measurement_18', 'sensor_measurement_19', 'sensor_measurement_20', 'sensor_measurement_21']) 
    flights = flights.groupby(['engine_no'])
    X_test3 = []
    y_test3 = []
    residuals = []
    features = []
    for key, item in flights:
        X_test3 = flights.get_group(key).iloc[:, 1:4]
        y_test3 = flights.get_group(key).iloc[:, 4:]
        sc3 = StandardScaler()
        X_test3 = sc3.fit_transform(X_test3)
        y_test3 = np.asarray(y_test3)
        y_pred3 = reduced_model.predict(X_test3)
        residuals.append(abs(np.array(y_pred3) - np.array(y_test3)))
        features.append([np.mean(residuals[-1])**2, np.std(residuals[-1]), np.mean(residuals[-1])])
    features = np.array(features)
    return features

# HMM Training

def hmm_train(features):

    gmmhmm = GMMHMM(n_components = 30, n_mix = 8)
    gmmhmm.startprob_ = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    l = np.identity(30)*0.95
    for i in range(l.shape[0] - 1):
        l[i, i + 1] = 0.05
    l[-1, -1] = 1
    gmmhmm.transmat_ = l
    gmmhmm.fit(features)
    preds = gmmhmm.predict(features)
    print(preds)

if __name__ == "__main__":
    
    saved_models_path = './saved_models/'
    data_path = 'train_FD004.txt'
    data = utils.load_data(data_path)
    data['RUL'] = 1
    gb = data.groupby(['engine_no'])
        
    process_db()
    
    # model = train_model()
    
    loaded_model = get_model('test.h5')
    
    # save_model(model,'test.h5')
    
    X_and_y, kmeans = perform_k_means()
    
    X_and_y_samples = reduce_samples(X_and_y, kmeans)
    
    reduced_model = get_model("test_reduced.h5")
    
    # save_model(reduced_model, 'test_reduced.h5')
    
    features = feature_extract(reduced_model)
    
    hmm_train(features)
