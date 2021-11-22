import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from generate_data import dataset, chars_from_ids, ids_from_chars
from model import model, OneStep


# Apply sparse categorical crossentropy loss function
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile with loss function and ADAM optimizer
model.compile(optimizer='adam', loss=loss)
print("Model compiled successfully")

# Directory to save the model checkpoints
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
print("Checkpoints configured")

# The number of epochs
EPOCHS = 2

# Train the model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
print("Training completed")

# Instantiate model object
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# Export the model
tf.saved_model.save(one_step_model, 'one_step')
print("Model saved")
