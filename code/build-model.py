import pandas as pd
import numpy as np

import onnx
import tensorflow as tf
import keras2onnx

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json

TRAINING_DATA = './training-data/scooby-doo-lines.csv'
CLASSES_JSON = './encoders/classes.json'
WORD_INDEX_JSON = './encoders/word_index.json'
TF_MODEL_FILE = './model/mystery-machine-learning.pb'
ONNX_MODEL_FILE = './model/mystery-machine-learning.onnx'

################################################################################
# Load: the data from storage and break it into features and labels. Features
# are the values used make predictions. For us, these are lines from the script.
# Labels are the thing to be predicted. For this example, that is the character.
#

# load the data from the file
df = pd.read_csv(TRAINING_DATA)

print()
print(f"Loaded data of shape {df.shape} from {TRAINING_DATA}")

# separate the features from the labels them one-dimensional
X_features = df.filter(items = ['line']).to_numpy().ravel()
y_labels = df.filter(items = ['character']).to_numpy().ravel()

print()
print(f"Split features and labels:")
print(f"  X (features) shape={np.shape(X_features)} type={X_features.dtype}")
print(f"  y (labels) shape={np.shape(y_labels)} type={y_labels.dtype}")

################################################################################
# Encode: Convert character names like 'Scooby-Doo', 'Shaggy Rogers', or 'Fred
# Jones' to integers.
#

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# sets up the encoder, defining a map of character names to integers
encoder = LabelEncoder()
encoder.fit(y_labels)

# save the classes for later user
classes = encoder.classes_

# save the classes to JSON for external use
with open(CLASSES_JSON, 'w') as f:
  f.write(json.dumps(classes.tolist(), indent = 2))
  f.close()

print()
print(f"Setting up label encoder:")
print(f"  Labels:  {classes}")
print(f"  Encoded: {encoder.transform(classes)}")

# encoded the labels using the encoder
y_labels = encoder.transform(y_labels)

print()
print(f"Encoded the labels:")
print(f"  X (features) shape={np.shape(X_features)} type={X_features.dtype}")
print(f"  y (labels) shape={np.shape(y_labels)} type={y_labels.dtype}")

# convert the labels from numbers to one-hot encoding
Y_labels = to_categorical(y_labels, len(classes))

print()
print(f"One-hot encoded the labels:")
print(f"  X (features) shape={np.shape(X_features)} type={X_features.dtype}")
print(f"  Y (labels) shape={np.shape(Y_labels)} type={Y_labels.dtype}")

################################################################################
# Tokenize: Convert lines like 'Right Raggy. Ret's Ro' to word vectors that we
# can do actual machine learning with.
#

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_VOCABULARY = 10000
MAX_LINE_LENGTH = 150

# build a tokenizer that can convert words to a sequence of integers
tokenizer = Tokenizer(num_words = MAX_VOCABULARY)
tokenizer.fit_on_texts(X_features)

# convert all the lines to an array of sequences
sequences = tokenizer.texts_to_sequences(X_features)

# pad the sequences so they are all the same length
X_features = pad_sequences(sequences, maxlen = MAX_LINE_LENGTH)

print()
print(f"Tokenized the features:")
print(f"  X (features) shape={np.shape(X_features)} type={X_features.dtype}")
print(f"  Y (labels) shape={np.shape(Y_labels)} type={Y_labels.dtype}")

# save this for later, it converts words to numbers
word_index = tokenizer.word_index

with open(WORD_INDEX_JSON, 'w') as f:
  f.write(json.dumps(word_index, indent = 2))
  f.close()

################################################################################
# Test/Train Split: Split the features and labels into test data and training
# data. The training data would be used to train the model. The test data will
# be used to validate the model.
# 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_labels,
  test_size = 0.2, random_state = 42)

print()
print(f"Split test and train data:")
print(f"  X_train shape={np.shape(X_train)} type={X_train.dtype}")
print(f"  Y_train shape={np.shape(Y_train)} type={Y_train.dtype}")
print(f"  X_test shape={np.shape(X_test)} type={X_test.dtype}")
print(f"  Y_test shape={np.shape(Y_test)} type={Y_test.dtype}")

################################################################################
# Build: Build up the layers of our model.
# 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

EMBEDDING_MATRIX_COLUMNS = 32

# it's a sequential model
model = Sequential(name = 'mystery-machine')

# add the embedding layer
model.add(
  Embedding(
    input_dim = MAX_VOCABULARY,
    output_dim = EMBEDDING_MATRIX_COLUMNS,
    input_length = MAX_LINE_LENGTH))

# add the RNN layer
model.add(SimpleRNN(units = EMBEDDING_MATRIX_COLUMNS))

# add an output layers
model.add(Dense(len(classes), activation = 'softmax'))

# compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# print the summary
print()
model.summary()

################################################################################
# Train: Train and evaluate the model.
#

# train the model
print()
model.fit(X_train, Y_train, epochs = 10, batch_size = 25)

# evaluate the model
print()
model.evaluate(X_test, Y_test)

################################################################################
# Predict: Make some predictions!
#

print()
print(f"Test Predictions:")

lines = [
  'jinkies',
  'like hang on scoob',
  'rokay raggy'
]

for line in lines:

  words = line.lower().split(' ')
  word_vector = [word_index[word] for word in words]
  pad_count = MAX_LINE_LENGTH - len(word_vector)

  sequence = np.array(word_vector)
  sequence = np.pad(sequence, (pad_count, 0),
    'constant', constant_values = (0))
  sequence = sequence.reshape(1, MAX_LINE_LENGTH)

  prediction = model.predict(sequence)
  encoded_class = np.argmax(prediction, axis = -1)
  decoded_class = encoder.inverse_transform(encoded_class)
  decoded_classes = np.stack([classes, prediction[0]], axis = 1)

  print()
  print(f"Line: '{line}' {word_vector} likely said by {decoded_class[0]} {encoded_class}")
  print("Sequence:")
  print(sequence)
  print()

  for index, (character, score) in enumerate(decoded_classes):
    print(f"{score:.5f} - {character} [{index}]")

  print()

################################################################################
# Save: Convert the model to ONNX and save.
#

print()
print(f"Saving an ONNX model:")

# convert
onnx_model = keras2onnx.convert_keras(model, model.name)

# save
keras2onnx.save_model(onnx_model, ONNX_MODEL_FILE)

################################################################################
# Save: Save the model to a frozen TensorFlow model.
#

from tensorflow import TensorSpec
from tensorflow.io import write_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

print()
print(f"Saving a frozen TensorFlow model:")

# convert the Keras model to a concrete function
spec = TensorSpec(shape = model.inputs[0].shape, dtype = model.inputs[0].dtype)
full_model = tf.function(lambda x: model(x)).get_concrete_function(spec)

# freeze that concrete function
frozen_func = convert_variables_to_constants_v2(full_model)
graph_def = frozen_func.graph.as_graph_def()

# save the frozen graph to disk
write_graph(
  graph_or_graph_def = graph_def,
  logdir = '.',
  name = TF_MODEL_FILE,
  as_text = False)

print(f"Frozen model input node:  {frozen_func.inputs}")
print(f"Frozen model output node: {frozen_func.outputs}")

print()
