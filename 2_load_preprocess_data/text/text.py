import collections
import pathlib
import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_datasets as tfds
import tensorflow_text as tf_text


data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset = utils.get_file(
    'stack_overflow_16k.tar.gz',
    data_url,
    untar=True,
    cache_dir='stack_overflow',
    cache_subdir=''
)
dataset_dir = pathlib.Path(dataset).parent

train_dir = dataset_dir/'train'
list(train_dir.iterdir())

sample_file = train_dir/'python/1755.txt'
with open(sample_file) as f:
    print(f.read())

batch_size, seed = 32, 42

"""
Training
"""

raw_train_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(10):
        print('Question: ', text_batch.numpy()[i][:100], '...')
        print('Label: ', label_batch.numpy()[i])

for i, label in enumerate(raw_train_ds.class_names):
    print("Label", i, "corresponds to", label)



"""
Validation
"""

raw_val_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)


test_dir = dataset_dir/'test'
raw_test_ds = preprocessing.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size
)

"""
TextVectorization
- Standardization
- Tokenization
- Vectorization
"""

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

binary_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary'
)

int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode=int,
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

train_text = raw_train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)
