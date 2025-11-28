from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 100
    dropout_rate: float = 0.5
    train_backbone: bool = False
    augmentation_layer: object = None 

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    use_mlflow: bool = True