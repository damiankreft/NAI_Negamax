from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers


dataset_path = "gs://cloud-training-demos/wine_quality/winequality-white.csv"

column_names = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar',
                'chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density',
                'pH','sulphates','alcohol','quality']

raw_dataframe = pd.read_csv(dataset_path, names=column_names, header = 0, 
                      na_values = " ", comment='\t',
                      sep=";", skipinitialspace=True)

raw_dataframe = raw_dataframe.astype(float)
raw_dataframe['quality'] = raw_dataframe['quality'].astype(int)
dataframe = raw_dataframe.copy()

dataframe.isna().sum()

dataframe.tail()

data_stats = dataframe.describe()
data_stats = data_stats.transpose()
data_stats

sns.pairplot(dataframe[["quality", "citric_acid", "residual_sugar", "alcohol"]], diag_kind="kde")

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(dataframe, epochs=10, shuffle=True, batch_size=64):
  dataframe = dataframe.copy()
  labels = tf.keras.utils.to_categorical(dataframe.pop('quality'), num_classes=11) #extracting the column which contains the training label
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.repeat(epochs).batch(batch_size)
  return ds

train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val, shuffle=False)
test_ds = df_to_dataset(test, shuffle=False)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of citric acid:', feature_batch['citric_acid'])
  print('A batch of quality:', label_batch )


feature_columns = []

fixed_acidity = tf.feature_column.numeric_column('fixed_acidity')
bucketized_fixed_acidity = tf.feature_column.bucketized_column(
    fixed_acidity, boundaries=[3., 5., 7., 9., 11., 13., 14.])
feature_columns.append(bucketized_fixed_acidity)

volatile_acidity = tf.feature_column.numeric_column('volatile_acidity')
bucketized_volatile_acidity = tf.feature_column.bucketized_column(
    volatile_acidity, boundaries=[0., 0.2, 0.4, 0.6, 0.8, 1.])
feature_columns.append(bucketized_volatile_acidity)

citric_acid = tf.feature_column.numeric_column('citric_acid')
bucketized_citric_acid = tf.feature_column.bucketized_column(
    citric_acid, boundaries=[0., 0.4, 0.7, 1.0, 1.3, 1.8])
feature_columns.append(bucketized_citric_acid)

residual_sugar = tf.feature_column.numeric_column('residual_sugar')
bucketized_residual_sugar = tf.feature_column.bucketized_column(
    residual_sugar, boundaries=[0.6, 10., 20., 30., 40., 50., 60., 70.])
feature_columns.append(bucketized_residual_sugar)

chlorides = tf.feature_column.numeric_column('chlorides')
bucketized_chlorides = tf.feature_column.bucketized_column(
    chlorides, boundaries=[0., 0.1, 0.2, 0.3, 0.4])
feature_columns.append(bucketized_chlorides)

free_sulfur_dioxide = tf.feature_column.numeric_column('free_sulfur_dioxide')
bucketized_free_sulfur_dioxide = tf.feature_column.bucketized_column(
    free_sulfur_dioxide, boundaries=[1., 50., 100., 150., 200., 250., 300.])
feature_columns.append(bucketized_free_sulfur_dioxide)

total_sulfur_dioxide = tf.feature_column.numeric_column('total_sulfur_dioxide')
bucketized_total_sulfur_dioxide = tf.feature_column.bucketized_column(
    total_sulfur_dioxide, boundaries=[9., 100., 200., 300., 400., 500.])
feature_columns.append(bucketized_total_sulfur_dioxide)

density = tf.feature_column.numeric_column('density')
bucketized_density = tf.feature_column.bucketized_column(
    density, boundaries=[0.9, 1.0, 1.1])
feature_columns.append(bucketized_density)

pH = tf.feature_column.numeric_column('pH')
bucketized_pH = tf.feature_column.bucketized_column(
    pH, boundaries=[2., 3., 4.])
feature_columns.append(bucketized_pH)

sulphates = tf.feature_column.numeric_column('sulphates')
bucketized_sulphates = tf.feature_column.bucketized_column(
    sulphates, boundaries=[0.2, 0.4, 0.7, 1.0, 1.1])
feature_columns.append(bucketized_sulphates)

alcohol = tf.feature_column.numeric_column('alcohol')
bucketized_alcohol = tf.feature_column.bucketized_column(
    alcohol, boundaries=[8., 9., 10., 11., 12., 13., 14.])
feature_columns.append(bucketized_alcohol)

feature_columns

# Create a feature layer from the feature columns

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(8, activation='relu'),
  layers.Dense(8, activation='relu'),
  layers.Dense(11, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=50)

predictions = model.predict(feature_batch)

classes_x = np.argmax(predictions,axis=1)
classes_x

len(classes_x)

predictions2 = model.predict(test_ds)
# predictions

classes_x2 = np.argmax(predictions2,axis=1)
len(classes_x2)

classes_x2