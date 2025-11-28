import numpy as np
import optuna
import mlflow
import tensorflow as tf
import logging
import json

from src.config_schema import ModelConfig, TrainConfig
from src.models import get_model
from src.ButterflyTrainer import ButterflyTrainer
from src.data import make_cv_datasets  
from src.augmentation import get_augment_layer

from src.config import (
    TRAIN_DIR,
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
    model_name: str,
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
        augmentation_layer=get_augment_layer(),
    )

    train_config = TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        use_mlflow=True,          
    )

    fold_datasets = make_cv_datasets(
        train_dir=TRAIN_DIR,
        n_splits=n_splits,
        batch_size=batch_size,
        image_size=input_shape[:2],
        random_state=RANDOM_STATE,
    )

    fold_accuracies = []

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params({
            "lr": lr,
            "dropout": dropout,
            "batch_size": batch_size,
            "train_backbone": train_backbone,
            "model_name": model_name,
            "input_shape": str(input_shape),
            "epochs": epochs,
            "n_splits": n_splits,
        })

        for fold_idx, (train_ds, val_ds) in enumerate(fold_datasets):
            logger.info(f"Trial {trial.number} | Fold {fold_idx+1}/{len(fold_datasets)}")

            model = get_model(
                model_name=model_config.model_name,
                input_shape=model_config.input_shape,
                num_classes=model_config.num_classes,
                dropout_rate=model_config.dropout_rate,
                train_backbone=model_config.train_backbone,
                augmentation_layer=model_config.augmentation_layer,
            )

            logger.info(f"Model '{model_name}' initialized. {model.name}")

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
    model_name: str = DEFAULT_MODEL_NAME,
):

    experiment_name = f"butterfly_optuna_{model_name}"
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"Starting Study: {experiment_name}")
    logger.info(f"Model: {model_name}, Shape: {input_shape}, Splits: {n_splits}, Epochs: {epochs}")

    study = optuna.create_study(
        study_name=experiment_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    with mlflow.start_run(run_name=f"study_{model_name}") as run:
        study.optimize(
            lambda t: objective(
                t,
                input_shape=input_shape,
                num_classes=num_classes,
                n_splits=n_splits,
                epochs=epochs,
                model_name=model_name,
            ),
            n_trials=n_trials,
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

        import matplotlib.pyplot as plt

        trial_numbers = [t.number for t in study.trials if t.value is not None]
        trial_values = [t.value for t in study.trials if t.value is not None]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(trial_numbers, trial_values, marker="o")
        ax.set_xlabel("Trial")
        ax.set_ylabel("CV mean val_accuracy")
        ax.set_title(f"Optuna trial performance ({model_name})")

        best_n = study.best_trial.number
        best_v = study.best_value
        ax.scatter([best_n], [best_v], s=80)

        mlflow.log_figure(fig, "trials_cv_mean_val_accuracy.png")
        plt.close(fig)

        logger.info(f"Logged trials comparison plot to MLflow run {run.info.run_id}")


if __name__ == "__main__":
    
    best_params = {
    "lr": 6.761040877813161e-05,
    "dropout": 0.20609185624518492,
    "batch_size": 16,
    "train_backbone": True,
}

# тимчасова обгортка без Optuna:
    def run_single(best_params):
        trial = optuna.trial.FixedTrial(best_params)
        return objective(
            trial,
            input_shape=DEFAULT_INPUT_SHAPE,
            num_classes=100,
            n_splits=1,
            epochs=10,  # 10
            model_name="efficientnet_b0",
        )
    run_single(best_params)
    # run_study()