import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization
from keras import regularizers
import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow_addons as tfa

original_df = pd.read_csv('datasets/WineQT.csv')
print(original_df.head())


original_df = original_df.drop('Id',axis =1)
print(original_df.columns)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])