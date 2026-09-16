
import sys
import os

# Make the "src" package (Preprocessing, Models, Training, Utils) importable
# regardless of the working directory this script is launched from.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
import json
import time
import datetime
import platform
import uuid
from Preprocessing import build_au_region_plan, extract_raw_landmark_features
import torchvision
import numpy as np
import sklearn  # Added for sklearn version

from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from Models import (SimpleCNN, LandmarkCNN, LandmarkResNet18, StandardCNN, SCNN_FC, ECF_CNN, SCNN_FMF, S3D_CNN, LSCNN_FC, LSCNN_FMF, setup_transfer_learning_model, count_trainable_parameters)
from Training.train import train_model
from Training.eval import evaluate_model, evaluate_classification_metrics
from Utils import plot_precision_recall_curve, plot_sample_predictions, plot_training_curves
# from comparison import (
#     load_all_experiments,
#     save_master_json,
#     run_comparison
# )
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support


def log(step, msg):
    print(f"\n[{step}] {msg}\n", flush=True)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run a facial emotion recognition experiment.")
parser.add_argument("--epochs", "-e", type=int, default=25, help="Number of training epochs.")
parser.add_argument("--model", "-m", type=str, default="cnn", choices=["cnn", "ResNet18", "StandardCNN", "SCNN_FC", "ECF_CNN", "SCNN_FMF", "S3D_CNN", "LSCNN_FC", "LSCNN_FMF"], help="Model to train")
parser.add_argument("--lr", "-lr", type=float, default=None, help="Learning rate. Overrides experiment default if provided.")
parser.add_argument("--batch_size", "-bs", type=int, default=64, help="Batch size for DataLoaders.")
parser.add_argument("--augment", "-aug", action="store_true", help="Enable data augmentation.")
parser.add_argument("--output_dir", "-o", type=str, default="results_separate", help="Directory to save experiment outputs (separate folder for organization).")
parser.add_argument("--optimizer", "-opt", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer to use.")
parser.add_argument("--scheduler", "-sch", type=str, default="ReduceLROnPlateau", choices=["None", "ReduceLROnPlateau", "CosineAnnealingLR"], help="Learning rate scheduler.")
parser.add_argument("--device", "-dev", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use.")
parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed.")
parser.add_argument("--early_stopping_patience", "-esp", type=int, default=10, help="Early stopping patience.")
parser.add_argument("--no_plot", action="store_true", help="Disable plotting.")
# parser.add_argument("--compare_experiments", "-comp", type=str, default="", help="Compare experiments.")
parser.add_argument("--use_landmarks", action="store_true", help="Train using landmark patches.")
parser.add_argument("--weight_decay", "-wd", type=float, default=1e-5, help="Weight decay.")
parser.add_argument("--dropout", "-do", type=float, default=0.4, help="Dropout rate.")
parser.add_argument("--num_landmarks", "-nl", type=int, default=7, help="Number of landmark patches.")
parser.add_argument("--use_au", "--with_au", dest="use_au", nargs="+", default=None, help="Use AU-based landmarks (e.g., AU2 AU5 AU12 AU15).")
parser.add_argument("--momentum", "-mom", type=float, default=0.9, help="Momentum for SGD.")
parser.add_argument("--freeze_backbone", action="store_true", help="Freeze pretrained backbone.")
parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")

args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# --- Global Experiment Log ---
experiment_log = {}
pipeline_start_time = time.time()
experiment_log["experiment_id"] = str(uuid.uuid4())
experiment_log["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
experiment_log["timing"] = {}
experiment_log["model_performance_summary"] = {}
experiment_log["output_info"] = {}

# --- Determine Device ---
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

# --- System and Framework Information ---
experiment_log["system_info"] = {
    "os": platform.system(),
    "node_name": platform.node(),
    "release": platform.release(),
    "version": platform.version(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "python_version": platform.python_version(),
    "device_used": str(device),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "framework_versions": {
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
}

# --- DEVICE + SEED ---
SEED = args.seed
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_classes = 7
dataset_name = "RAF-DB"
experiment_log["dataset_info"] = {
    "name": dataset_name,
    "num_classes": num_classes,
    "random_seed": SEED,
    "dataset_path": "/kaggle/input/datasets/tanishmittal/rafdb"
}

# Corrected paths
IMG_DIR = os.path.join(experiment_log["dataset_info"]["dataset_path"], "dataset", "aligned")

log("1/6", "Loading CSV files")
dataset_loading_start_time = time.time()
train_csv = pd.read_csv("/kaggle/input/datasets/tanishmittal/rafdb/dataset/train_labels.csv")
test_csv = pd.read_csv("/kaggle/input/datasets/tanishmittal/rafdb/dataset/test_labels.csv")
all_data = pd.concat([train_csv, test_csv], ignore_index=True)
dataset_loading_end_time = time.time()
experiment_log["timing"]["dataset_loading_time"] = dataset_loading_end_time - dataset_loading_start_time

data_preprocessing_start_time = time.time()
train_df, temp_df = train_test_split(
    all_data,
    test_size=0.30,
    stratify=all_data["label"],
    random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)

train_df["encoded_label"] = train_df["label"] - 1
val_df["encoded_label"] = val_df["label"] - 1
test_df["encoded_label"] = test_df["label"] - 1

experiment_log["dataset_info"].update({
    "train_samples": len(train_df),
    "validation_samples": len(val_df),
    "test_samples": len(test_df),
    "train_val_test_split_ratio": "70/15/15 (approx)"
})


class FaceEmotionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, use_landmarks=False, num_landmarks=7, use_au=False, au_list=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.use_landmarks = use_landmarks
        self.num_landmarks = num_landmarks
        self.use_au = use_au
        self.au_list = au_list

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        filename_from_df = self.dataframe.iloc[idx]['image']

        if filename_from_df.startswith('train_'):
            subdir = 'train'
            actual_image_folder = 'train_images'
        elif filename_from_df.startswith('test_'):
            subdir = 'test'
            actual_image_folder = 'test_images'
        else:
            raise ValueError(f"Unexpected image filename: {filename_from_df}")

        full_image_path = os.path.join(self.img_dir, subdir, actual_image_folder, filename_from_df)
        image = Image.open(full_image_path).convert('RGB')
        label = self.dataframe.iloc[idx]['label'] - 1

        if self.use_landmarks:
            landmark_patches = extract_raw_landmark_features(
                image,
                patch_size=48,
                num_landmarks=self.num_landmarks,
                with_au=self.use_au,
                au_list=self.au_list
            )
            return (landmark_patches, torch.tensor(label, dtype=torch.long))

        if self.transform:
            image = self.transform(image)

        return (image, torch.tensor(label, dtype=torch.long))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

log("2/6", "Creating Datasets")

resolved_au_codes = None
resolved_num_landmarks = args.num_landmarks
use_au_regions = bool(args.use_au)

if use_au_regions and not args.use_landmarks:
    raise ValueError("use_au requires use_landmarks!")

if use_au_regions and args.model not in ["cnn", "ResNet18", "SCNN_FC", "ECF_CNN", "SCNN_FMF", "S3D_CNN", "LSCNN_FC", "LSCNN_FMF"]:
    raise ValueError(f"Model {args.model} doesn't support AU-based landmarks!")

if use_au_regions:
    resolved_au_codes = []
    for raw_value in args.use_au:
        resolved_au_codes.extend([item.strip() for item in str(raw_value).split(",") if item.strip()])
    resolved_au_plan = build_au_region_plan(resolved_au_codes)
    resolved_num_landmarks = len(resolved_au_plan["landmark_groups"])
    if resolved_num_landmarks <= 0:
        raise ValueError("AU-based plan resulted in 0 landmarks!")

# Build AU/landmark identifier for filenames
if use_au_regions:
    au_str = "-".join(resolved_au_codes)
    landmark_identifier = f"AU-{au_str}"
else:
    if args.use_landmarks:
        landmark_identifier = f"{resolved_num_landmarks}-landmarks"
    else:
        landmark_identifier = "full-image"

# Build full filename prefix
full_model_name = args.model.replace("_", "-")
filename_prefix = f"{landmark_identifier}_{full_model_name}_{dataset_name}"

train_dataset_aug = FaceEmotionDataset(
    train_df,
    IMG_DIR,
    transform=transform_train,
    use_landmarks=args.use_landmarks,
    num_landmarks=resolved_num_landmarks,
    use_au=use_au_regions,
    au_list=resolved_au_codes
)

train_dataset_no_aug = FaceEmotionDataset(
    train_df,
    IMG_DIR,
    transform=transform_test,
    use_landmarks=args.use_landmarks,
    num_landmarks=resolved_num_landmarks,
    use_au=use_au_regions,
    au_list=resolved_au_codes
)

val_dataset = FaceEmotionDataset(
    val_df,
    IMG_DIR,
    transform=transform_test,
    use_landmarks=args.use_landmarks,
    num_landmarks=resolved_num_landmarks,
    use_au=use_au_regions,
    au_list=resolved_au_codes
)

test_dataset = FaceEmotionDataset(
    test_df,
    IMG_DIR,
    transform=transform_test,
    use_landmarks=args.use_landmarks,
    num_landmarks=resolved_num_landmarks,
    use_au=use_au_regions,
    au_list=resolved_au_codes
)
data_preprocessing_end_time = time.time()
experiment_log["timing"]["data_preprocessing_time"] = data_preprocessing_end_time - data_preprocessing_start_time

log("3/6", "Creating dataloaders")

train_loader_aug = DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

# --- Comparison logic commented out ---
# if args.compare_experiments:
#     log("Comparing Experiments", ...)
#     ...

# --- Determine experiment parameters ---
current_lr = args.lr
current_augment = args.augment
current_train_loader = (train_loader_aug if current_augment else train_loader_no_aug)

model_init_start_time = time.time()

if args.model in ["cnn", "LandmarkCNN", "SCNN_FC", "ECF_CNN", "SCNN_FMF", "S3D_CNN", "LSCNN_FC", "LSCNN_FMF"]:
    current_lr = 0.1 if args.lr is None else args.lr
elif args.model == "ResNet18":
    current_lr = 0.0001 if args.lr is None else args.lr
else:
    current_lr = 0.001 if args.lr is None else args.lr

if args.model == "cnn":
    if args.use_landmarks:
        model = LandmarkCNN(
            num_classes=num_classes,
            dropout=args.dropout,
            num_landmarks=resolved_num_landmarks
        ).to(device)
    else:
        model = SimpleCNN(num_classes=num_classes).to(device)
    experiment_log["model_config"] = {"model_name": args.model}
elif args.model == "StandardCNN":
    model = StandardCNN(
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    experiment_log["model_config"] = {"model_name": args.model}
elif args.model in ["SCNN_FC", "ECF_CNN", "SCNN_FMF", "S3D_CNN", "LSCNN_FC", "LSCNN_FMF"]:
    if not args.use_landmarks:
        raise ValueError(f"{args.model} requires landmark input!")
    model_class = {
        "SCNN_FC": SCNN_FC,
        "ECF_CNN": ECF_CNN,
        "SCNN_FMF": SCNN_FMF,
        "S3D_CNN": S3D_CNN,
        "LSCNN_FC": LSCNN_FC,
        "LSCNN_FMF": LSCNN_FMF
    }[args.model]
    model = model_class(
        num_landmarks=resolved_num_landmarks,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    experiment_log["model_config"] = {"model_name": args.model}
elif args.model == "ResNet18":
    if args.use_landmarks:
        model = LandmarkResNet18(
            num_classes=num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout,
            num_landmarks=resolved_num_landmarks
        ).to(device)
    else:
        model = setup_transfer_learning_model(
            model_name="resnet18",
            num_classes=num_classes,
            device=device,
            freeze_features=True
        )
    experiment_log["model_config"] = {"model_name": args.model}

total_params = count_total_parameters(model)
trainable_params = count_trainable_parameters(model)

experiment_log["model_config"].update({
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "batch_size": args.batch_size,
    "learning_rate": current_lr,
    "optimizer_name": args.optimizer,
    "loss_function": "CrossEntropyLoss",
    "scheduler_information": args.scheduler,
    "data_augmentation_configuration": "RandomHorizontalFlip/Rotation/ColorJitter" if current_augment else "None",
    "data_augmentation_enabled": current_augment,
    "early_stopping_patience": args.early_stopping_patience,
    "dropout": args.dropout,
    "weight_decay": args.weight_decay,
    "pretrained": args.pretrained,
    "freeze_backbone": args.freeze_backbone,
    "num_landmarks": resolved_num_landmarks,
    "with_au": use_au_regions,
    "landmark_identifier": landmark_identifier,
    "dataset_name": dataset_name
})

model_init_end_time = time.time()
experiment_log["timing"]["model_initialization_time"] = model_init_end_time - model_init_start_time

criterion = nn.CrossEntropyLoss()

optimizer = None
if args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=args.weight_decay)
elif args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    raise ValueError(f"Optimizer {args.optimizer} not supported!")

scheduler = None
if args.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
elif args.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

print("\n--- DEBUG: Sample Forward Pass ---")
sample_inputs, sample_labels = next(iter(current_train_loader))
sample_inputs = sample_inputs.to(device)
with torch.no_grad():
    sample_output = model(sample_inputs)
print(f"Input shape: {sample_inputs.shape}")
print(f"Output shape: {sample_output.shape}")
print("--- END DEBUG ---\n")

log("4/6", f"Starting training for {args.model}")
training_start_time = time.time()
train_losses, val_losses, train_accuracies, val_accuracies, epoch_times, best_model = train_model(
    model,
    current_train_loader,
    val_loader,
    criterion,
    optimizer,
    device=device,
    scheduler=scheduler,
    num_epochs=args.epochs,
    patience=args.early_stopping_patience
)
training_end_time = time.time()
actual_num_epochs = len(train_losses)
total_training_time = training_end_time - training_start_time
average_epoch_time = total_training_time / actual_num_epochs
experiment_log["timing"]["total_training_time"] = total_training_time
experiment_log["timing"]["average_epoch_training_time"] = average_epoch_time

# --- NEW: Store metrics as simple lists ---
experiment_log["train_losses"] = train_losses
experiment_log["val_losses"] = val_losses
experiment_log["train_accuracies"] = train_accuracies
experiment_log["val_accuracies"] = val_accuracies
experiment_log["epoch_times"] = epoch_times
experiment_log["epochs_run"] = actual_num_epochs
experiment_log["epochs_total"] = args.epochs
experiment_log["epochs_run_over_total"] = f"{actual_num_epochs}/{args.epochs}"

test_loss, test_accuracy = evaluate_model(best_model, test_loader, criterion, device)
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

log("5/6", "Evaluating model")
evaluation_start_time = time.time()
metrics_dict = evaluate_classification_metrics(best_model, test_loader, device, num_classes, args.output_dir, filename_prefix) or {}
experiment_log["evaluation_metrics"] = metrics_dict
experiment_log["test_metrics"] = {
    "top1_accuracy": metrics_dict.get("top1_accuracy", 0),
    "top2_accuracy": metrics_dict.get("top2_accuracy", 0),
    "precision_macro": metrics_dict.get("precision_macro", 0),
    "precision_weighted": metrics_dict.get("precision_weighted", 0),
    "recall_macro": metrics_dict.get("recall_macro", 0),
    "recall_weighted": metrics_dict.get("recall_weighted", 0),
    "f1_macro": metrics_dict.get("f1_macro", 0),
    "f1_weighted": metrics_dict.get("f1_weighted", 0),
    "confusion_matrix": metrics_dict.get("confusion_matrix", [])
}

print("\n--- Training Summary ---")
print(f"Total training time: {total_training_time:.2f} sec")
print(f"Average epoch time: {average_epoch_time:.2f} sec")

print("\n--- Test Results ---")
print(f"Top-1 Acc: {metrics_dict.get('top1_accuracy', 0):.2f}%")
print(f"Top-2 Acc: {metrics_dict.get('top2_accuracy', 0):.2f}%")
evaluation_end_time = time.time()
experiment_log["timing"]["evaluation_time"] = evaluation_end_time - evaluation_start_time

log("6/6", "Generating plots")
class_names = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

if not args.no_plot:
    plot_training_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        output_dir=args.output_dir,
        filename_prefix=filename_prefix
    )

    plot_precision_recall_curve(
        best_model,
        test_loader,
        device,
        num_classes,
        class_names,
        output_dir=args.output_dir,
        filename_prefix=filename_prefix
    )

    plot_sample_predictions(
        best_model,
        test_loader,
        class_names,
        device,
        output_dir=args.output_dir,
        filename_prefix=filename_prefix
    )

log("7/6", "Saving model and results")
model_saving_start_time = time.time()
model_filename = f"Checkpoint_{filename_prefix}.pt"
model_save_path = os.path.join(args.output_dir, model_filename)
torch.save(best_model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")
model_saving_end_time = time.time()
experiment_log["timing"]["model_saving_time"] = model_saving_end_time - model_saving_start_time

best_val_acc = max(val_accuracies) if val_accuracies else 0.0
experiment_log["model_performance_summary"] = {
    "best_validation_accuracy": best_val_acc,
    "test_accuracy": test_accuracy,
    "model_name": args.model
}

pipeline_end_time = time.time()
experiment_log["timing"]["complete_pipeline_execution_time"] = pipeline_end_time - pipeline_start_time

# --- Save JSON Log ---
json_filename = f"ExperimentLog_{filename_prefix}.json"
json_log_path = os.path.join(args.output_dir, json_filename)
with open(json_log_path, "w") as f:
    json.dump(experiment_log, f, indent=4)
print(f"Experiment log saved to: {json_log_path}")

# --- Save Master JSON (commented out) ---
# master_json_path = os.path.join(args.output_dir, "all_experiments.json")
# all_experiments = load_all_experiments(master_json_path)
# all_experiments.append(experiment_log)
# save_master_json(all_experiments, master_json_path)

experiment_log["output_info"] = {
    "saved_model_path": model_save_path,
    "json_log_path": json_log_path,
    "csv_log_path": ""  # To be set below
}

# --- Research-Grade CSV Log (separate file per experiment) ---
csv_filename = f"Results_{filename_prefix}.csv"
csv_path = os.path.join(args.output_dir, csv_filename)
csv_data = {
    "Experiment ID": experiment_log["experiment_id"],
    "Timestamp": experiment_log["timestamp"],
    "Number of Classes": experiment_log["dataset_info"]["num_classes"],
    "Train Samples": experiment_log["dataset_info"]["train_samples"],
    "Validation Samples": experiment_log["dataset_info"]["validation_samples"],
    "Test Samples": experiment_log["dataset_info"]["test_samples"],
    "Model Name": args.model,
    "Total Parameters": total_params,
    "Trainable Parameters": trainable_params,
    "Input Size": str(sample_inputs.shape),
    "Batch Size": args.batch_size,
    "Learning Rate": current_lr,
    "Optimizer": args.optimizer,
    "Loss Function": "CrossEntropyLoss",
    "Epochs Run / Total": f"{actual_num_epochs}/{args.epochs}",
    "Early Stopping Patience": args.early_stopping_patience,
    "Scheduler": args.scheduler,
    "Data Augmentation Enabled": current_augment,
    "Best Validation Accuracy": best_val_acc,
    "Final Test Accuracy": test_accuracy,
    "Precision (Macro)": metrics_dict.get("precision_macro", "N/A"),
    "Recall (Macro)": metrics_dict.get("recall_macro", "N/A"),
    "F1 Score (Macro)": metrics_dict.get("f1_macro", "N/A"),
    "ROC-AUC Score": metrics_dict.get("roc_auc_score", "N/A"),
    "Average Epoch Time (sec)": average_epoch_time,
    "Total Training Time (sec)": total_training_time,
    "Total Pipeline Time (sec)": pipeline_end_time - pipeline_start_time,
    "Landmark Identifier": landmark_identifier
}

results_df = pd.DataFrame([csv_data])
results_df.to_csv(csv_path, index=False)
print(f"Experiment results CSV saved to: {csv_path}")
experiment_log["output_info"]["csv_log_path"] = csv_path

print("\n" + "=" * 60)
print("All tasks completed successfully!")
print("=" * 60)
