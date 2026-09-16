import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    output_dir=".",
    filename_prefix="TrainingCurves_7-landmarks_ModelName_RAF-DB"
):
    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{filename_prefix} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_filename = f"{output_dir}/LossCurve_{filename_prefix}.png"
    plt.savefig(loss_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {loss_filename}")

    # Plot Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title(f"{filename_prefix} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_filename = f"{output_dir}/AccuracyCurve_{filename_prefix}.png"
    plt.savefig(acc_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy curve to {acc_filename}")


# Commented out comparison functions for future use
# def compare_models(
#     val_loss_model1,
#     val_acc_model1,
#     val_loss_model2,
#     val_acc_model2,
#     name1="Custom CNN",
#     name2="ResNet34"
# ):
# 
#     plt.figure(figsize=(12,5))
# 
#     plt.subplot(1,2,1)
# 
#     plt.plot(val_loss_model1,label=f"{name1}")
#     plt.plot(val_loss_model2,label=f"{name2}")
# 
#     plt.title("Validation Loss Comparison")
# 
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
# 
#     plt.legend()
# 
#     plt.subplot(1,2,2)
# 
#     plt.plot(val_acc_model1,label=f"{name1}")
#     plt.plot(val_acc_model2,label=f"{name2}")
# 
#     plt.title("Validation Accuracy Comparison")
# 
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy (%)")
# 
#     plt.legend()
# 
#     plt.tight_layout()
# 
#     plt.savefig("model_comparison.png")
# 
#     plt.show()
#     plt.close()
# 
# def plot_augmentation_comparison(
#     train_losses_aug,
#     val_losses_aug,
#     train_losses_no_aug,
#     val_losses_no_aug,
#     train_acc_aug,
#     val_acc_aug,
#     train_acc_no_aug,
#     val_acc_no_aug,
#     filename="augmentation_comparison.png"
# ):
# 
#     plt.figure(figsize=(12,5))
# 
#     plt.subplot(1,2,1)
# 
#     plt.plot(train_losses_aug,label="Train Loss Aug")
#     plt.plot(val_losses_aug,label="Val Loss Aug")
# 
#     plt.plot(train_losses_no_aug,label="Train Loss No Aug")
#     plt.plot(val_losses_no_aug,label="Val Loss No Aug")
# 
#     plt.title("Augmented vs Non-Augmented Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
# 
#     plt.subplot(1,2,2)
# 
#     plt.plot(train_acc_aug,label="Train Acc Aug")
#     plt.plot(val_acc_aug,label="Val Acc Aug")
# 
#     plt.plot(train_acc_no_aug,label="Train Acc No Aug")
#     plt.plot(val_acc_no_aug,label="Val Acc No Aug")
# 
#     plt.title("Augmented vs Non-Augmented Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy (%)")
#     plt.legend()
# 
#     plt.tight_layout()
# 
#     plt.savefig(filename)
# 
#     plt.close()
# 
# def plot_lr_comparison(
#     val_loss_lr1,
#     val_acc_lr1,
#     val_loss_lr2,
#     val_acc_lr2,
#     filename="lr_comparison.png"
# ):
# 
#     plt.figure(figsize=(12,5))
# 
#     plt.subplot(1,2,1)
# 
#     plt.plot(val_loss_lr1,label="LR = 0.001")
#     plt.plot(val_loss_lr2,label="LR = 0.0001")
# 
#     plt.title("Validation Loss Comparison")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
# 
#     plt.subplot(1,2,2)
# 
#     plt.plot(val_acc_lr1,label="LR = 0.001")
#     plt.plot(val_acc_lr2,label="LR = 0.0001")
# 
#     plt.title("Validation Accuracy Comparison")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy (%)")
#     plt.legend()
# 
#     plt.tight_layout()
# 
#     plt.savefig(filename)
# 
#     plt.close()
# 
# def plot_final_accuracy_comparison(
#     accuracies,
#     labels,
#     filename="final_accuracy_comparison.png"
# ):
# 
#     plt.figure(figsize=(8,5))
# 
#     plt.bar(labels, accuracies)
# 
#     plt.ylabel("Accuracy (%)")
# 
#     plt.title("Final Test Accuracy Comparison")
# 
#     plt.xticks(rotation=20)
# 
#     plt.tight_layout()
# 
#     plt.savefig(filename)
# 
#     plt.close()

def plot_sample_predictions(
    model,
    test_loader,
    class_names,
    device,
    num_images=9,
    output_dir=".",
    filename_prefix="SamplePredictions_7-landmarks_ModelName_RAF-DB"
):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(10,10))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for j in range(images.size(0)):
                plt.subplot(3,3,images_shown+1)
                img_tensor = images[j].cpu()

                if img_tensor.ndim == 4:
                    img_tensor = img_tensor[0]

                img = img_tensor.permute(1,2,0).numpy()
                img = (
                    img * np.array([0.229, 0.224, 0.225])
                    + np.array([0.485, 0.456, 0.406])
                )
                img = np.clip(img, 0, 1)

                plt.imshow(img)
                plt.title(
                    f"P: {class_names[preds[j].item()]}\n"
                    f"T: {class_names[labels[j].item()]}"
                )
                plt.axis("off")

                images_shown += 1

                if images_shown == num_images:
                    plt.tight_layout()
                    filename = f"{output_dir}/SamplePredictions_{filename_prefix}.png"
                    plt.savefig(filename, dpi=300, bbox_inches="tight")
                    print(f"Saved sample predictions to {filename}")
                    plt.show()
                    plt.close()
                    return
