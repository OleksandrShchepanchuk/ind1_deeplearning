import tensorflow as tf
from tensorflow.keras import layers, models

def create_alexnet(
    input_shape=(224, 224, 3),
    num_classes=100,
    dropout_rate=0.5,
    augmentation_layer=None,
    train_backbone=True, 
    **kwargs
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    # x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = layers.Conv2D(96, kernel_size=11, strides=4, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(256, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="AlexNet")

def create_vggnet_custom(
    input_shape=(224, 224, 3),
    num_classes=100,
    dropout_rate=0.5,
    augmentation_layer=None,
    train_backbone=True, 
    **kwargs
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    for filters in [64, 128, 256, 512, 512]:
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="VGGNet_Custom")

def create_lenet(
    input_shape=(32, 32, 1),
    num_classes=10,
    dropout_rate=0.0,
    augmentation_layer=None,
    train_backbone=True, # Unused
    **kwargs
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)
    
    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = layers.Conv2D(6, kernel_size=5, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, kernel_size=5, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="LeNet")