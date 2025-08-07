def evaluate_model(model, test_ds, param_count=None):
    # Evaluate using Keras .evaluate (matches compile metrics order)
    loss, auc, precision, recall, accuracy, iou = model.evaluate(test_ds, steps=5)
    print(" Test Evaluation Metrics:")
    print(f"Loss: {loss:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"IoU: {iou:.4f}")
    if param_count:
        print(f"Model Size: {param_count / (1024 ** 2):.2f} MB")

    # Save all metrics, including efficiency ones
    return {
        "loss": loss, "auc": auc, "precision": precision,
        "recall": recall, "accuracy": accuracy, "iou": iou, "params": param_count
    }
