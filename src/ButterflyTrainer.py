import tensorflow as tf
import mlflow
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from .config_schema import TrainConfig
from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

class ButterflyTrainer:
    def __init__(self, model, train_ds, val_ds, train_config: TrainConfig):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.config = train_config

    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            
            loss="sparse_categorical_crossentropy", 
            
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_acc")
            ]
        )
        

    def train(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]

        if self.config.use_mlflow:
            mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

        logger.info(f"Starting training for {self.config.epochs} epochs...")
        
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks
        )
        
        if self.config.use_advanced_metrics:
            self.evaluate_and_log_advanced_metrics()
            
        return history

    def evaluate_and_log_advanced_metrics(self):

        logger.info("Calculating advanced metrics (F1, Confusion Matrix)...")
        
        y_true = []
        y_pred = []
        
        for img_batch, label_batch in self.val_ds:
            preds = self.model.predict(img_batch, verbose=0)
            y_true.extend(label_batch.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
            
        report = classification_report(y_true, y_pred, output_dict=True)
        f1_macro = report['macro avg']['f1-score']
        mlflow.log_metric("val_f1_macro", f1_macro)
        
        f1_weighted = report['weighted avg']['f1-score']
        mlflow.log_metric("val_f1_weighted", f1_weighted)

        logger.info(f"Validation F1 Macro: {f1_macro:.4f}")

        text_report = classification_report(y_true, y_pred)
        
        run = mlflow.active_run()
        if run:
            run_id = run.info.run_id
            
            report_filename = f"report_{self.model.name}_{run_id}.txt"
        else:
            report_filename = f"report_{self.model.name}.txt"

        report_path = PROJECT_ROOT / report_filename

        with open(report_path, "w") as f:
            f.write(text_report)
            
        mlflow.log_artifact(report_path)
        logger.info(f"Full classification report saved to MLflow artifacts.")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)