
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def evaluate_model(model, test_ds, albinated_Feature,param_count=None):
    # Evaluate using Keras .evaluate (matches compile metrics order)
    loss, auc, precision, recall, accuracy, iou,pr_Auc = model.evaluate(test_ds, steps=5)
    print(" Test Evaluation Metrics:")
    print(f"Loss: {loss:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"PR-AUC:{pr_Auc:.4f}")
    plot_results(auc, precision, recall, accuracy, iou,pr_Auc,albinated_Feature)
    plot_radar_chart(auc, precision, recall, accuracy, iou,pr_Auc,albinated_Feature)
    if param_count:
        print(f"Model Size: {param_count / (1024 ** 2):.2f} MB")

    # Save all metrics, including efficiency ones
    return {
        "loss": loss, "auc": auc, "precision": precision,
        "recall": recall, "accuracy": accuracy, "iou": iou, "params": param_count
    }

def plot_results(auc, precision, recall, accuracy, iou, pr_Auc,albinated_Feature):

    proportions = [auc, precision, recall, accuracy, iou, pr_Auc]
    labels = ['AUC', 'Precision', 'Recall', 'Accuracy', 'IoU', 'PR_Auc']
    N = len(proportions)

    # Extend proportions for triangulation
    proportions = np.append(proportions, 1)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.append(np.sin(theta), 0)
    y = np.append(np.cos(theta), 0)

    # Triangles for polygon
    triangles = [[N, i, (i + 1) % N] for i in range(N)]
    triang_backgr = tri.Triangulation(x, y, triangles)
    triang_foregr = tri.Triangulation(x * proportions, y * proportions, triangles)

    # Colors
    cmap = plt.cm.rainbow_r
    colors = np.linspace(0, 1, N + 1)

    # Plot
    plt.tripcolor(triang_backgr, colors, cmap=cmap, shading='gouraud', alpha=0.4)
    plt.tripcolor(triang_foregr, colors, cmap=cmap, shading='gouraud', alpha=0.8)
    plt.triplot(triang_backgr, color='white', lw=2)

    # Add labels + values
    for label, value, xi, yi in zip(labels, proportions[:-1], x, y):
        plt.text(
            xi * 1.15, yi * 1.15,
            f"{label}\n{value:.2f}",  # label + value on 2 lines
            ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
            va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center',
            fontsize=10, fontweight='bold'
        )
    plt.title(f" {albinated_Feature}", size=14, y=1.1)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.show()
'''
def plot_radar_chart(auc, precision, recall, accuracy, iou, pr_Auc,albinated_Feature,model_name="Model"):

    # Extract metric names and values
    labels = ['AUC', 'Precision', 'Recall', 'Accuracy', 'IoU', 'PR_Auc']
    values = [auc, precision, recall, accuracy, iou, pr_Auc]
    
    # Repeat the first value to close the circle
    values += values[:1]
    N = len(labels)
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    
    # Draw outline
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
    ax.fill(angles, values, alpha=0.25)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Set radial limits (0–1 since metrics are probabilities)
    ax.set_ylim(0, 1)
    
    # Add grid and legend
    plt.title(f"Radar Chart: {albinated_Feature}", size=14, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plt.show()
'''

def plot_radar_chart(auc, precision, recall, accuracy, iou, pr_Auc,model_scores_dict,model_name="Model", title="Model Performance Comparison"):
    labels = ['AUC', 'Precision', 'Recall', 'Accuracy', 'IoU', 'PR_Auc']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the circle

    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    model_scores_dict[model_name] = [auc, precision, recall, accuracy, iou, pr_Auc]

    # Plot each model
    for model_name, scores in model_scores_dict.items():
        scores = scores + scores[:1]  # close the polygon
        ax.plot(angles, scores, linewidth=2, label=model_name)
        ax.fill(angles, scores, alpha=0.25)

    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)  # adjust if scores are not normalized to [0,1]

    # Title and legend
    ax.set_title(title, va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.show()

