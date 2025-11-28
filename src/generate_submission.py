import os
import pandas as pd
import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_submission(
    model_path: str,
    test_csv_path: str,
    images_root_dir: str,
    class_names_path: str,
    output_path: str = "submission.csv",
    batch_size: int = 32,
    image_size: tuple = (224, 224)
):
    
  
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class list not found at {class_names_path}")
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    logger.info(f"✅ Loaded {len(class_names)} classes.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Reading test data from {test_csv_path}...")
    df = pd.read_csv(test_csv_path)

    df['absolute_path'] = df['path'].apply(lambda x: os.path.join(images_root_dir, x))

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="absolute_path",
        y_col=None,              
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,        
        shuffle=False,           
        validate_filenames=False 
    )

    logger.info(f"Running inference on {len(df)} images...")
    predictions = model.predict(test_generator, verbose=1)

    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = [class_names[i] for i in predicted_indices]

    submission = pd.DataFrame({
        'id': df['id'],
        'label': predicted_labels
    })
    
    submission.to_csv(output_path, index=False)
    logger.info(f"✅ Success! Submission saved to '{output_path}'")

if __name__ == "__main__":
    generate_submission(
        model_path="../models/convexnet_base2_version_01drop.keras",
        test_csv_path="../data/test.csv",
        images_root_dir="../data",   
        class_names_path="../data/classes.txt",
        output_path="submission_final.csv"
    )