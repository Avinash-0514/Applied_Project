import os
from tensorflow.keras.utils import plot_model

def summarize_and_plot(model, name):
        print(f"\n🔍 Model Summary: {name}")
        model.summary()
        param_count = model.count_params()
        print(f"Total Trainable Parameters: {param_count}")

        # Update the base directory for saving diagrams
        base_save_dir = ""
        diagram_save_path = os.path.join(base_save_dir, "architecture_diagrams")
        os.makedirs(diagram_save_path, exist_ok=True) # Create directory

        # Save model diagram as PNG
        try:
            # Use the updated diagram path
            plot_model(model, to_file=os.path.join(diagram_save_path, f"{name}_architecture.png"), show_shapes=True)
            print(f"Model diagram saved as {os.path.join(diagram_save_path, f'{name}_architecture.png')}")
        except Exception as e:
            print(f"Could not save model diagram: {e}")

        return param_count