import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from common import (EPOCHS)

def train_model(model, train_ds, val_ds, name, feature_index):
        # Update the base directory for saving models and logs
        base_save_dir = ""
        model_save_path = os.path.join(base_save_dir, "models", name,feature_index)
        log_dir_path = os.path.join(base_save_dir, "logs", name,feature_index)

        # Create directories if they don't exist
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_dir_path, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_save_path, "best_model.keras"), # Use filepath instead of first argument
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                monitor="val_loss",
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir_path) # Use the updated log path
        ]
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=5,
            validation_steps=5,
            callbacks=callbacks
        )
        return history

