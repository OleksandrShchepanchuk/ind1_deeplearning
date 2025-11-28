import tensorflow as tf
from tensorflow.keras import layers, models, applications

def create_vgg16_imagenet(
    input_shape=(224, 224, 3),
    num_classes=100,
    dropout_rate=0.5,
    train_backbone=False,
    augmentation_layer=None,
    **kwargs
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = applications.vgg16.preprocess_input(x)

    base_model = applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = train_backbone

    x = base_model(x, training=train_backbone)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="VGG16_ImageNet")

def create_efficientnet_imagenet(
    input_shape=(224, 224, 3),
    num_classes=100,
    dropout_rate=0.5,
    train_backbone=False,
    augmentation_layer=None,
    model_name="efficientnet_b0", 
    **kwargs
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = applications.efficientnet.preprocess_input(x)

    if "b3" in model_name:
        base_model = applications.EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    elif "b4" in model_name:
        base_model = applications.EfficientNetB4(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        base_model = applications.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

    base_model.trainable = train_backbone

    x = base_model(x, training=train_backbone)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return models.Model(inputs, outputs, name=f"EfficientNet_ImageNet")


def create_convnext_imagenet(
    input_shape=(224, 224, 3),
    num_classes=100,
    dropout_rate=0.3, 
    train_backbone=False,
    augmentation_layer=None,
    model_name="convnext_base", 
    **kwargs
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    if "tiny" in model_name:
        base_model = applications.ConvNeXtTiny(weights="imagenet", include_top=False, input_tensor=x)
    elif "small" in model_name:
        base_model = applications.ConvNeXtSmall(weights="imagenet", include_top=False, input_tensor=x)
    elif "large" in model_name:
        base_model = applications.ConvNeXtLarge(weights="imagenet", include_top=False, input_tensor=x)
    else:
        base_model = applications.ConvNeXtBase(weights="imagenet", include_top=False, input_tensor=x)

    base_model.trainable = train_backbone

    x = base_model.output
    
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return models.Model(inputs, outputs, name=f"ConvNeXt_{model_name}")