import os
from tensorflow import keras
from model.model import weighted_bce, iou_metric  

models_dir = "/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Combine_Feature_Model/models"

# Recursively search for all `.keras` files
model_files = []
for root, dirs, files in os.walk(models_dir):
    for f in files:
        if f.endswith(".keras"):
            model_files.append(os.path.join(root, f))

print(f" Found {len(model_files)} .keras files")


model_list = []
failed_models = []

# Try loading each model
for model_path in model_files:
    try:
        print(f"Loading model: {model_path}")
        model = keras.models.load_model(
            model_path,
            custom_objects={"weighted_bce": weighted_bce, "iou_metric": iou_metric}
        )
        model_list.append(model)
    except Exception as e:
        print(f" Failed to load {model_path}: {e}")
        failed_models.append(model_path)

print(f"\n Successfully loaded {len(model_list)} models.")

def ensemble_predict(models, x):
    preds = [m.predict(x) for m in models]  # get predictions from each model
    return sum(preds) / len(preds)   