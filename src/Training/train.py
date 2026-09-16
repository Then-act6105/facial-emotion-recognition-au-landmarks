import torch
import copy
import time


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    num_epochs=25,
    patience=10
):

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []
    epoch_times = [] # Added to store epoch times

    best_val_loss = float('inf')
    epochs_no_improve = 0

    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time() # Start timing for the epoch

        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)

            total_train += labels.size(0)

            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train

        epoch_train_accuracy = 100 * correct_train / total_train

        train_losses.append(epoch_train_loss)

        train_accuracies.append(epoch_train_accuracy)

        # Validation

        model.eval()

        val_loss = 0.0

        correct_val = 0

        total_val = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)

                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)

                total_val += labels.size(0)

                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val

        epoch_val_accuracy = 100 * correct_val / total_val

        val_losses.append(epoch_val_loss)

        val_accuracies.append(epoch_val_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_accuracy:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_accuracy:.2f}%"
        )

        if scheduler is not None:

            scheduler.step(epoch_val_loss)

        epoch_end_time = time.time()
        epoch_times.append(epoch_end_time - epoch_start_time) # Store epoch time

        if epoch_val_loss < best_val_loss:

            best_val_loss = epoch_val_loss

            epochs_no_improve = 0

            best_model_wts = copy.deepcopy(model.state_dict())

        else:

            epochs_no_improve += 1

            if epochs_no_improve >= patience:

                print(
                    f"Early stopping triggered after "
                    f"{epoch+1} epochs."
                )

                break

    model.load_state_dict(best_model_wts)

    print("Training complete.")

    return (
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        epoch_times, # Returned epoch times
        model
    )
