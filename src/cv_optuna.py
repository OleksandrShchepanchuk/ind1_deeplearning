import numpy as np
import optuna
import mlflow
import tensorflow as tf
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import json

from src.config_schema import ModelConfig, TrainConfig
from src.models import get_model
from src.ButterflyTrainer import ButterflyTrainer
from src.data import build_train_index_from_directory, make_train_dataset_from_df
from src.augmentation import get_augment_layer

from src.config import (
    INPUT_SHAPE as DEFAULT_INPUT_SHAPE,
    NUM_CLASSES as DEFAULT_NUM_CLASSES,
    N_SPLITS as DEFAULT_N_SPLITS,
    EPOCHS_OPTUNA as DEFAULT_EPOCHS_OPTUNA,
    BEST_MODEL_NAME as DEFAULT_MODEL_NAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_N_TRIALS = 20

def objective(
    trial: optuna.Trial,
    input_shape: tuple,
    num_classes: int,
    n_splits: int,
    epochs: int,
    model_name: str
) -> float:

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_backbone = trial.suggest_categorical("train_backbone", [False, True])
    
    model_config = ModelConfig(
        model_name=model_name,
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout,
        train_backbone=train_backbone,
        augmentation_layer=get_augment_layer() 
    )

    train_config = TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        use_mlflow=False 
    )

    df, label2id = build_train_index_from_directory()
    X = df["path"].values
    y = df["label_id"].values

    if n_splits > 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(X, y))
    else:
        logger.info(" n_splits=1 detected. Using simple 80/20 Hold-Out validation instead of CV.")
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), 
            test_size=0.2, 
            stratify=y, 
            random_state=RANDOM_STATE
        )
        splits = [(train_idx, val_idx)] 
    
    fold_accuracies = []

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params({
            "lr": lr,
            "dropout": dropout,
            "batch_size": batch_size,
            "train_backbone": train_backbone,
            "model_name": model_name,
            "input_shape": str(input_shape),
            "epochs": epochs
        })

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Trial {trial.number} | Fold {fold_idx+1}/{len(splits)}")
            
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_ds = make_train_dataset_from_df(
                train_df, 
                batch_size=batch_size, 
                shuffle=True,
                image_size=input_shape[:2]
            )
            val_ds = make_train_dataset_from_df(
                val_df, 
                batch_size=batch_size, 
                shuffle=False, 
                augment_layer=None,
                image_size=input_shape[:2]
            )

            model = get_model(
                model_name=model_config.model_name,
                input_shape=model_config.input_shape,
                num_classes=model_config.num_classes,
                dropout_rate=model_config.dropout_rate,
                train_backbone=model_config.train_backbone,
                augmentation_layer=model_config.augmentation_layer
            )
            logger.info(f"Model '{model_name}' initialized. {model.name}")
            print(f"Model '{model_name}' initialized. {model.name}")
            model.summary()
            trainer = ButterflyTrainer(model, train_ds, val_ds, train_config)
            trainer.compile()
            history = trainer.train()

            val_acc = history.history["val_accuracy"][-1]
            fold_accuracies.append(val_acc)
            mlflow.log_metric(f"val_accuracy_fold_{fold_idx}", val_acc)

            del model, trainer, history, train_ds, val_ds
            tf.keras.backend.clear_session()

        mean_acc = float(np.mean(fold_accuracies))
        mlflow.log_metric("cv_mean_val_accuracy", mean_acc)

    return mean_acc

def run_study(
    n_trials: int = DEFAULT_N_TRIALS,
    input_shape: tuple = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    n_splits: int = DEFAULT_N_SPLITS,
    epochs: int = DEFAULT_EPOCHS_OPTUNA,
    model_name: str = DEFAULT_MODEL_NAME
):

    experiment_name = f"butterfly_optuna_{model_name}"
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"Starting Study: {experiment_name}")
    logger.info(f"Model: {model_name}, Shape: {input_shape}, Splits: {n_splits}, Epochs: {epochs}")

    study = optuna.create_study(
        study_name=experiment_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner() 
    )
    
    study.optimize(
        lambda t: objective(
            t, 
            input_shape=input_shape, 
            num_classes=num_classes, 
            n_splits=n_splits, 
            epochs=epochs, 
            model_name=model_name
        ), 
        n_trials=n_trials
    )

    logger.info("------------------------------------------------")
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best CV Accuracy: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    logger.info("------------------------------------------------")

    best_params_path = f"best_params_{model_name}.json"
    
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    mlflow.log_artifact(best_params_path)
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_cv_accuracy", study.best_value)
    
    if mlflow.active_run():
        logger.info(f"Successfully logged best params to MLflow run '{mlflow.active_run().info.run_id}'")

if __name__ == "__main__":
    run_study()