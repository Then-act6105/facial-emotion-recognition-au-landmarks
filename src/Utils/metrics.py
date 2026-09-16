import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def plot_precision_recall_curve(
    model,
    test_loader,
    device,
    num_classes,
    class_names,
    output_dir=".",
    filename_prefix="PrecisionRecall_7-landmarks_ModelName_RAF-DB"
):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())

            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)

    all_labels = np.concatenate(all_labels)

    y_true = label_binarize(
        all_labels,
        classes=np.arange(num_classes)
    )

    plt.figure(figsize=(10,7))

    for i in range(num_classes):

        precision, recall, _ = precision_recall_curve(
            y_true[:, i],
            all_probs[:, i]
        )

        ap = average_precision_score(
            y_true[:, i],
            all_probs[:, i]
        )

        plt.plot(
            recall,
            precision,
            label=f"{class_names[i]} (AP={ap:.2f})"
        )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title(f"{filename_prefix} - Precision Recall Curve")

    plt.legend()

    plt.grid(True)

    filename = f"{output_dir}/PrecisionRecall_{filename_prefix}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved precision recall curve to {filename}")

    plt.show()
    plt.close()
