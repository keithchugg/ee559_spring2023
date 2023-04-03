import tensorflow as tf

# Source: Tensorflow documentation https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
# modify the code to suit your needs

model = tf.keras.Model()
model.compile(metrics=['accuracy'])

EPOCHS = 10
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_filepath,
save_weights_only=True,
monitor='val_accuracy',
mode='max',
save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen so far.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)