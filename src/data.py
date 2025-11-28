import tensorflow as tf
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

AUTOTUNE = tf.data.AUTOTUNE
RANDOM_STATE = 42  # той самий, що й у cv_optuna


def build_train_index_from_directory(
    train_dir: Path | str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Будує DataFrame зі шляхами до зображень і label_id.
    train_dir: коренева папка з підпапками-класами.
    """
    train_dir = Path(train_dir)

    records = []
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        label = class_dir.name
        # Matches .jpg, .jpeg, .JPG
        for img_path in sorted(class_dir.glob("*.[jJ][pP]*[gG]")):
            records.append({"path": str(img_path), "label": label})

    df = pd.DataFrame(records)
    
    classes = sorted(df["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label2id)
    
    return df, label2id


def load_image(
    path: str,
    image_size: Tuple[int, int],
) -> tf.Tensor:
    """
    Читає й ресайзить зображення до image_size.
    Нормалізацію (поділ на 255, preprocess_input) робиш де завгодно вище.
    """
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
    image_size: Tuple[int, int] = (224, 224),
    cache: bool = False,
) -> tf.data.Dataset:
    """
    Створює tf.data.Dataset з DataFrame.
    df має містити колонки "path" та "label_id".
    """
    paths = df["path"].values
    labels = df["label_id"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: (load_image(p, image_size), l),
        num_parallel_calls=AUTOTUNE,
    )

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size)

    if augment_layer is not None:
        ds = ds.map(
            lambda x, y: (augment_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.prefetch(AUTOTUNE)

    return ds


def make_cv_datasets(
    train_dir: Path | str,
    batch_size: int,
    image_size: Tuple[int, int] = (224, 224),
    n_splits: int = 0,
    random_state: int = RANDOM_STATE,
) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:

    df, label2id = build_train_index_from_directory(train_dir)
    X = df["path"].values
    y = df["label_id"].values

    if n_splits > 1:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(X, y))
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=0.2,
            stratify=y,
            random_state=random_state,
        )
        splits = [(train_idx, val_idx)]

    datasets: List[Tuple[tf.data.Dataset, tf.data.Dataset]] = []

    for train_idx, val_idx in splits:
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = make_train_dataset_from_df(
            train_df,
            batch_size=batch_size,
            shuffle=True,
            image_size=image_size,
        )

        val_ds = make_train_dataset_from_df(
            val_df,
            batch_size=batch_size,
            shuffle=False,
            augment_layer=None,
            image_size=image_size,
        )

        datasets.append((train_ds, val_ds))

    return datasets


def load_test_dataframe(
    data_dir: Path | str,
    test_csv_name: str,
    test_path_col: str,
    test_images_root: Path | str,
) -> pd.DataFrame:

    data_dir = Path(data_dir)
    test_images_root = Path(test_images_root)

    csv_path = data_dir / test_csv_name
    df = pd.read_csv(csv_path)
    df[test_path_col] = df[test_path_col].apply(
        lambda p: str(test_images_root / p)
    )
    return df


def make_test_dataset(
    df: pd.DataFrame,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    test_path_col: str = "path",
) -> tf.data.Dataset:

    paths = df[test_path_col].values
    
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda p: load_image(p, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    
    return ds


def load_train_val_from_directory(
    train_dir: Path | str,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 123,
    image_size: Tuple[int, int] = (224, 224),
):

    df, _ = build_train_index_from_directory(train_dir)
    
    val_df = df.sample(frac=val_split, random_state=seed)
    train_df = df.drop(val_df.index)
    
    train_ds = make_train_dataset_from_df(
        train_df,
        batch_size=batch_size,
        shuffle=True,
        image_size=image_size,
    )
    val_ds = make_train_dataset_from_df(
        val_df,
        batch_size=batch_size,
        shuffle=False,
        image_size=image_size,
    )
    
    class_names = sorted(df["label"].unique())
    
    return train_ds, val_ds, class_names
