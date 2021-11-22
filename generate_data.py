import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import get_file

import numpy as np
import os
import time


# Download the Shakespeare dataset
path_to_file = get_file(
        'shakespeare.txt', 
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
        )
print("Data downloaded")

# Read the data and decode it
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# To get the length of the dataset
print(f'Length of text: {len(text)} characters')

# To get the unique characters in the data
vocab = sorted(set(text))
print(f'There are {len(vocab)} unique characters in the text')

# Convert each character into a numeric id
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab), mask_token=None)

# Convert numeric id into characters
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# Function to extract string from the characters converted from the numeric ids
def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Generate ids for all the tokens in the text
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

# Convert the text vector into a stream of character indices
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# To convert the individual characters to sequences of the desired size
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

# Function to take a sequence as input, duplicate and shift it to align the input and label for each timestep
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000

# Generate the training dataset
dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

print("Data generated")
