import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)


def evaluate_model(model, data_loader, criterion,device):
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = running_loss / total
    accuracy = 100 * correct / total
    return average_loss, accuracy


def evaluate_classification_metrics(model, data_loader, device, num_classes, output_dir=".", filename_prefix="7-landmarks_ModelName_RAF-DB"):
    model.eval() # Set model to evaluation mode
    all_labels = []
    all_predictions = []
    all_probs = []
    top2_correct = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            _, top2_preds = torch.topk(outputs.data, k=2, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Check top-2 accuracy
            for i in range(labels.size(0)):
                if labels[i] in top2_preds[i]:
                    top2_correct += 1

    metrics = {}
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)

    # Top-1 and Top-2 accuracy
    top1_accuracy = 100 * np.sum(all_labels_np == all_predictions_np) / len(all_labels_np)
    top2_accuracy = 100 * top2_correct / len(all_labels_np)
    metrics["top1_accuracy"] = top1_accuracy
    metrics["top2_accuracy"] = top2_accuracy

    # Precision, Recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels_np, all_predictions_np, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels_np, all_predictions_np, average='weighted'
    )
    metrics["precision_macro"] = precision_macro
    metrics["recall_macro"] = recall_macro
    metrics["f1_macro"] = f1_macro
    metrics["precision_weighted"] = precision_weighted
    metrics["recall_weighted"] = recall_weighted
    metrics["f1_weighted"] = f1_weighted

    report = classification_report(all_labels, all_predictions, digits=4, output_dict=True)
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_predictions, digits=4))
    metrics["classification_report"] = report

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_predictions)
    metrics["confusion_matrix"] = cm.tolist() # Convert numpy array to list for JSON serialization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{filename_prefix} - Confusion Matrix")
    cm_filename = f"{output_dir}/ConfusionMatrix_{filename_prefix}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix to {cm_filename}")
    plt.close()

    # Calculate AUC ROC (one-vs-rest)
    # Convert labels to one-hot encoding for AUC calculation
    all_labels_one_hot = np.eye(num_classes)[all_labels]
    all_probs_np = np.array(all_probs)

    # Check if there's more than one class present in the labels for AUC calculation
    if len(np.unique(all_labels)) > 1:

        auc_score_macro = roc_auc_score(
            all_labels_one_hot,
            all_probs_np,
            average='macro',
            multi_class='ovr'
        )

        print(f"\nAUC ROC (Macro Avg, One-vs-Rest): {auc_score_macro:.4f}")
        metrics["roc_auc_score"] = auc_score_macro

        plt.figure(figsize=(8,6))

        for i in range(num_classes):

            fpr, tpr, _ = roc_curve(
                all_labels_one_hot[:, i],
                all_probs_np[:, i]
            )

            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f"Class {i} (AUC={roc_auc:.2f})"
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{filename_prefix} - ROC Curve (One-vs-Rest)")
        plt.legend()

        roc_filename = f"{output_dir}/ROCCurve_{filename_prefix}.png"
        plt.savefig(roc_filename, dpi=300, bbox_inches="tight")
        print(f"Saved ROC curve to {roc_filename}")
        plt.close()

    else:
        print("\nAUC ROC cannot be calculated for a single class in the dataset.")
        metrics["roc_auc_score"] = "N/A"

    return metrics
