from generate_data import vocab, ids_from_chars

import tensorflow as tf


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# Define the model
class MyModel(tf.keras.Model):
  
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    # The input layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    # The RNN layer
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    # The output layer
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

# Instantiate an RNN model object
model = MyModel(
    # vocabulary size must match the `StringLookup` layers
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)
