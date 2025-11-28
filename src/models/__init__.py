# src/models/__init__.py
from .custom import create_alexnet, create_lenet, create_vggnet_custom
# Import the new function here -----------------------v
from .imagenet import create_vgg16_imagenet, create_efficientnet_imagenet, create_convnext_imagenet
from .hub_backbones import create_effnetv2_hub

MODEL_REGISTRY = {
    "alexnet": create_alexnet,
    "lenet": create_lenet,
    "vggnet_custom": create_vggnet_custom,
    "vgg16_imagenet": create_vgg16_imagenet,
    "efficientnet_b0": create_efficientnet_imagenet,
    "efficientnet_b3": create_efficientnet_imagenet,
    
    "convnext_tiny": create_convnext_imagenet,
    "convnext_small": create_convnext_imagenet,
    "convnext_base": create_convnext_imagenet,
    
    "effnetv2_b0_21k": create_effnetv2_hub
}


def get_model(
    model_name: str, 
    input_shape=(224, 224, 3), 
    num_classes=100, 
    dropout_rate=0.5, 
    train_backbone=False,
    augmentation_layer=None
):
    """
    Factory function to instantiate any model with a unified API.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")

    model_fn = MODEL_REGISTRY[model_name]
    
    
    return model_fn(
        model_name=model_name, 
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        train_backbone=train_backbone,
        augmentation_layer=augmentation_layer
    )