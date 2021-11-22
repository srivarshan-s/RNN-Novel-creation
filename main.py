# Setup
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as numpy
import os
import time

# Download dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the data
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(f'Length of text: {len(text)} characters')
print(text[:250])
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters are present in the text')

# Process the text
example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)
ids_from_chars = preprocessing.StringLookup(
        vocabulary=list(vocab), 
        mask_token=None
)
ids = ids_from_chars(chars)
print(ids)
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(),
    invert=True,
    mask_token=None
)
chars = chars_from_ids(ids)
chars
tf.strings.reduce_join(chars, axis=-1).numpy()
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


