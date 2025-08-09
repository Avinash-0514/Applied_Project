from utils.utils import (summarize_and_plot)
from evaluation.eval import (evaluate_model)
from training.train import (train_model)
from data.preprocessing import(dataset_split_function)
from common import (training,testing,validation,BATCH_SIZE)
from model.model import(build_multi_kernel_cnn_model)
from common import (input_features,BATCH_SIZE)

def main():
    model_name = "MultiKernelCNN"
   # nn_model = build_multi_kernel_cnn_model()
    ensemble_flag = True
    feature_combination =[]

    if ensemble_flag:
        all_results = {}
        for feature_index in input_features:
            print(f"\nTraining model for Feature {feature_index} + Fire Mask")
            selected_feats = [feature_index] if ensemble_flag else input_features
            # Call dataset_split_function for current feature
            conv_train_ds, conv_val_ds, conv_test_ds = dataset_split_function(
                training, validation, testing,
                batch_size=BATCH_SIZE,
                isFire=True,
                selected_features=[feature_index]
            )

            nn_model = build_multi_kernel_cnn_model(len(selected_feats) + 1)
            param_count = summarize_and_plot(nn_model, f"{model_name}_F{feature_index}")
            train_model(nn_model, conv_train_ds, conv_val_ds, f"{model_name}_F{feature_index}",str(feature_index))
            all_results[f"{model_name}_F{feature_index}"] = evaluate_model(nn_model, conv_test_ds, param_count)

        print("\nFinal Results for all single-feature models:")
        for name, metrics in all_results.items():
            print(f"{name}: {metrics}")
    else:
        # Train one model with combined features
        conv_train_ds, conv_val_ds, conv_test_ds = dataset_split_function(
            training, validation, testing,
            batch_size=BATCH_SIZE,
            ensemble_flag=False,
            selected_features=feature_combination
        )

        nn_model = build_multi_kernel_cnn_model()
        param_count = summarize_and_plot(nn_model, f"{model_name}_Combined")
        train_model(nn_model, conv_train_ds, conv_val_ds, f"{model_name}_Combined")
        final_metrics = evaluate_model(nn_model, conv_test_ds, param_count)

        print(f"\nFinal Results for Combined Features Model: {final_metrics}")

 

if __name__ == '__main__':
    main()


