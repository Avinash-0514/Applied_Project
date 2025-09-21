from utils.utils import (summarize_and_plot)
from evaluation.eval import (evaluate_model)
from training.train import (train_model)
from data.preprocessing import(dataset_split_function)
from common import (training,testing,validation,BATCH_SIZE)
from model.model import(build_multi_kernel_cnn_model)
from common import (input_features,BATCH_SIZE)


def main ():
    # Get Layer & Filter data
    model_name = "MultiKernelCNN"
    Isprev_feat_append_flag = bool(int(input("Enter the True(1) to Append prev_Feat with all feature else False(0)")))
    num_layers = int(input("Enter the Number of layers (eg., 1,2,3)"))
    base_filters = int(input("Enter the Base filters(eg., 32,64,128)"))
    isCustomModel = bool(int(input("Enter the True (1) or False (0) To train all feature")))
    enum_feat =('tmmn','NDVI','population','elevation','vs','pdsi','pr','tmmx','sph','th','erc')
    if isCustomModel != True:
        single_feature = bool(int(input("Enter the Ture(1) or False (0) To train Single feature")))
        
    model_type = ""

    
    if isCustomModel:
        model_type = "AllFeature"
        # Training with all features with the fire mask.
        conv_train_ds, conv_val_ds, conv_test_ds = dataset_split_function(
            training, 
            validation,
            testing,
            prev_feat_append_flag = Isprev_feat_append_flag,
            batch_size=BATCH_SIZE,
            multiple_input = False,
            selected_features = input_features
            )
        nn_model = build_multi_kernel_cnn_model(
            #num_input_channels = len(input_features)*2,
            num_input_channels= len(input_features) * 2 if Isprev_feat_append_flag else len(input_features) + 1,
            layerCount = num_layers,
            filter_size = base_filters
            )
        #count_parameter = summarize_and_plot(nn_model, f"{model_name}_Combined")
        #train_model(nn_model, conv_train_ds, conv_val_ds,model_type, f"{model_name}_Combined","All")
        count_parameter = summarize_and_plot(nn_model, f"{model_name}_Combined")
        train_model(nn_model, conv_train_ds, conv_val_ds,model_type, f"{model_name}_Combined","All")
        matrics = evaluate_model(nn_model, conv_test_ds, count_parameter)
        print("\nFinal Results for All Combined Features Model: {matrics}")
    else:
        if single_feature:
            model_type ="SingleFeature"
            all_results ={}
            #Training with single feature with Fire mask
            for feature_index in input_features:
                conv_train_ds, conv_val_ds,conv_test_ds = dataset_split_function(
                    training,
                    validation,
                    testing,
                    prev_feat_append_flag = Isprev_feat_append_flag,
                    batch_size=BATCH_SIZE,
                    multiple_input = False,
                    selected_features= [feature_index]

                )
                nn_model = build_multi_kernel_cnn_model(
                    num_input_channels = 2,
                    layerCount = num_layers,
                    filter_size = base_filters
                    )
                #count_parameter = summarize_and_plot(nn_model,f"{model_name}_Single")
                #train_model(nn_model, conv_train_ds, conv_val_ds,model_type, f"{model_name}_Single","Single")
                count_parameter = summarize_and_plot(nn_model,f"{feature_index}_Single")
                train_model(nn_model, conv_train_ds, conv_val_ds,model_type, f"{feature_index}_Single","Single")
                all_results[f"{model_name}_F{feature_index}"] = evaluate_model(nn_model, conv_test_ds, count_parameter)
            
            print("\nFinal Results for all single-feature models:")
            for name, metrics in all_results.items():
                print(f"{name}: {metrics}")
        else:
            # Custom feature based on the matrices
            x = list(enumerate(enum_feat,start=1))
            print(x)

            model_type = "CombinedFeature"
            enum_feat =('tmmn','NDVI','population','elevation','vs','pdsi','pr','tmmx','sph','th','erc')

            custom_features =[]
            x = list(enumerate(enum_feat,start=1))
            features = input("Enter the top features:")
            value = [int(d) for d in features]

            for i in value:
                print(i)
                custom_features.append(input_features[i-1])
            albinated_Feature = "-".join(custom_features)


            conv_train_ds,conv_val_ds,conv_test_ds = dataset_split_function(
                    training,
                    validation,
                    testing,
                    prev_feat_append_flag = Isprev_feat_append_flag,
                    batch_size=BATCH_SIZE,
                    multiple_input = False,
                    selected_features=custom_features
                )
            nn_model = build_multi_kernel_cnn_model(
                    #num_input_channels = len(custom_features)*2,
                    num_input_channels= len(custom_features) * 2 if Isprev_feat_append_flag else len(custom_features) + 1,
                    layerCount = num_layers,
                    filter_size = base_filters
                )
            #count_parameter = summarize_and_plot(nn_model,f"{model_name}_Custom")
            #train_model(nn_model, conv_train_ds, conv_val_ds,model_type, f"{model_name}_Custom","Combined")
            count_parameter = summarize_and_plot(nn_model,f"{model_name}_Custom")
            train_model(nn_model, conv_train_ds, conv_val_ds,model_type, f"{albinated_Feature}_Custom","Combined")
            matrics = evaluate_model(nn_model, conv_test_ds, albinated_Feature,count_parameter) 
            print("\nFinal Results for All Combined Features Model: {matrics}")   
            
if __name__ == '__main__':
    main()


