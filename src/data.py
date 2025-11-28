import tensorflow as tf
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict

from .config import (
    TRAIN_DIR,
    TEST_CSV_NAME,
    TEST_ID_COL,
    TEST_PATH_COL,
    TEST_IMAGES_ROOT,
    INPUT_SHAPE,
    DATA_DIR
)

AUTOTUNE = tf.data.AUTOTUNE

def build_train_index_from_directory() -> Tuple[pd.DataFrame, Dict[str, int]]:

    records = []
    class_dirs = sorted([d for d in Path(TRAIN_DIR).iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        label = class_dir.name
        for img_path in sorted(class_dir.glob("*.[jJ][pP]*[gG]")): # Matches .jpg, .jpeg, .JPG
            records.append({"path": str(img_path), "label": label})

    df = pd.DataFrame(records)
    
    classes = sorted(df["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label2id)
    
    return df, label2id

def load_image(path: str, image_size: Tuple[int, int] = INPUT_SHAPE[:2]) -> tf.Tensor:

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) 
    return img

def make_train_dataset_from_df(
    df: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = True,
    augment_layer: Optional[tf.keras.layers.Layer] = None,
    image_size: Tuple[int, int] = INPUT_SHAPE[:2],
    cache: bool = False
) -> tf.data.Dataset:

    paths = df["path"].values
    labels = df["label_id"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

   
   
    ds = ds.map(lambda p, l: (load_image(p, image_size), l), num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size)


    if augment_layer is not None:
        ds = ds.map(lambda x, y: (augment_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(AUTOTUNE)

    return ds

def load_test_dataframe() -> pd.DataFrame:
    csv_path = DATA_DIR / TEST_CSV_NAME
    df = pd.read_csv(csv_path)
    df[TEST_PATH_COL] = df[TEST_PATH_COL].apply(
        lambda p: str(TEST_IMAGES_ROOT / p)
    )
    return df

def make_test_dataset(
    df: pd.DataFrame,
    batch_size: int = 32,
    image_size: Tuple[int, int] = INPUT_SHAPE[:2]
) -> tf.data.Dataset:
    paths = df[TEST_PATH_COL].values
    
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: load_image(p, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    
    return ds

def load_train_val_from_directory(
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 123,
):

    df, _ = build_train_index_from_directory()
    
    val_df = df.sample(frac=val_split, random_state=seed)
    train_df = df.drop(val_df.index)
    
    train_ds = make_train_dataset_from_df(train_df, batch_size=batch_size, shuffle=True)
    val_ds = make_train_dataset_from_df(val_df, batch_size=batch_size, shuffle=False)
    
    class_names = sorted(df["label"].unique())
    
    return train_ds, val_ds, class_names