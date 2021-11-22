import tensorflow as tf

from model import model


# Apply sparse categorical crossentropy loss function
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile with loss function and ADAM optimizer
model.compile(optimizer='adam', loss=loss)
print("Model compiled successfully")
