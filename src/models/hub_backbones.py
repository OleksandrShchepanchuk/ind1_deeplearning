import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_hub as hub

HUB_URLS = {
    "effnetv2_b0_21k": "https://www.kaggle.com/models/google/efficientnet-v2/TensorFlow2/imagenet21k-b0-feature-vector/1",
}

def create_effnetv2_hub(
    input_shape=(224, 224, 3),
    num_classes=100,
    dropout_rate=0.3,
    train_backbone=False,
    augmentation_layer=None,
    model_name="effnetv2_b0_21k",
    **kwargs
):
   
    if model_name not in HUB_URLS:
        hub_url = HUB_URLS["effnetv2_b0_21k"]
    else:
        hub_url = HUB_URLS[model_name]

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    x = layers.Rescaling(1./255)(x)
    if augmentation_layer is not None:
        x = augmentation_layer(x)



    hub_layer = hub.KerasLayer(
        hub_url,
        trainable=train_backbone,
        name="effnetv2_backbone"
    )
    
    x = hub_layer(x)
    
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="EfficientNetV2_Hub")