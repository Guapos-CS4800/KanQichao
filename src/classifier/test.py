import tensorflow as tf


reconstructed_model = tf.keras.models.load_model("my_model.h5")
reconstructed_model.summary()