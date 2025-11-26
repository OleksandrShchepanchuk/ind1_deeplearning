import tensorflow as tf
from tensorflow.keras import layers, models, applications

def get_pretrained_vgg16_model(input_shape=(224, 224, 3), num_classes=100, augmentation_layer=None):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = inputs
    if augmentation_layer is not None:
        x = augmentation_layer(x)
    
    x = applications.vgg16.preprocess_input(x)
    
    base_model = applications.VGG16(
        weights='imagenet', 
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False 
    
    x = base_model(x, training=False)
    
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="VGG16_Butterfly_Classifier")
    return model