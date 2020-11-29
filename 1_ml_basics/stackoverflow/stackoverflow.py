# 2.3.1

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

url = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

#load data
dataset = tf.keras.utils.get_file("stack_overflow_16k.tar.gz", url, untar=True, cache_dir='./downloads/', cache_subdir='stackoverflow')
dataset_dir = os.path.join(os.path.dirname(dataset))

print(dataset_dir, "test", os.getcwd())
training_dir = os.path.join(dataset_dir, 'train')

"""
Set the training batch size and seed for randomization and transformation
"""
batch_size = 15000
seed = 275

"""
Preprocess the training data
"""
raw_training_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'downloads/stackoverflow/train',
    batch_size=batch_size,
    validation_split=0.4,
    subset='training',
    seed=seed
)

""" Preprocess the validation data set """
raw_validation_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'downloads/stackoverflow/train',
    batch_size=batch_size,
    validation_split=0.4,
    subset='validation',
    seed=seed
)

"""
Preprocess the test data
"""
raw_testing_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'downloads/stackoverflow/test',
    batch_size=batch_size
)

"""
Set the maximum feature length aka vocabulary size and the output sequence length
for the TextVectorization layer 
"""

max_features, sequence_length = 10000, 250

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

training_text = raw_training_ds.map(lambda x, y: x)
vectorize_layer.adapt(training_text)

text_batch, label_batch = next(iter(raw_training_ds))
first_review, first_label = text_batch[0], label_batch[0]

training_data_cnt = 200
o_loop_cnt = 10

for test_batch, label_batch in raw_training_ds:

    o_loop_cnt -= 1
    for i in range(training_data_cnt):
        print("Review", text_batch[i])
        print("Label", raw_training_ds.class_names[label_batch[i]])
        print("\n")
        #print("Vectorized Review", vectorize_text(first_review, first_label))
    print(" " * 20, "\n", "=" * 20, "\n", " " * 20)

    if o_loop_cnt == 0: break


training_ds = raw_training_ds.map(vectorize_text)
validation_ds = raw_validation_ds.map(vectorize_text)
testing_ds = raw_testing_ds.map(vectorize_text)

AUTOTUNE = tf.data.experimental.AUTOTUNE

training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dimensions = 15

model = tf.keras.Sequential(
    [
        layers.Embedding(max_features + 1, embedding_dimensions),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4)
    ]
)

model.summary()

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

epochs = 20
history = model.fit(
    training_ds,
    validation_data=validation_ds,
    epochs=epochs
)

loss, accuracy = model.evaluate(testing_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

loss, accuracy = export_model.evaluate(raw_testing_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

examples = [
    "windows",
    ""
]

def print_predictions(predictions):
    for prediction in predictions:
        print(
            prediction,
            np.argmax(prediction),
            raw_training_ds.class_names[np.argmax(prediction)]
        )

    print("\n")

predictions_0 = export_model.predict(x=examples, use_multiprocessing=True)
print_predictions(predictions_0)