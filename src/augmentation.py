import tensorflow as tf

def get_augment_layer() -> tf.keras.layers.Layer:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1), 
            tf.keras.layers.RandomZoom(0.1),     
            tf.keras.layers.RandomContrast(0.2), 
            tf.keras.layers.RandomBrightness([-0.2, 0.2]) 
        ],
        name="data_augmentation",
    )