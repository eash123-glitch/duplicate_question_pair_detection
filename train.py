# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import os

# def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
#     """
#     Train the model and save the training loss plot with a name based on the model class.

#     Args:
#         model: The neural network model to train.
#         train_loader: DataLoader for the training data.
#         criterion: Loss function.
#         optimizer: Optimizer.
#         device: Device to train the model on (CPU or GPU).
#         epochs: Number of training epochs.
#         save_dir: Directory to save the loss plot.
#     """
#     model.to(device)
#     train_losses = []
#     save_dir = './plots'
#     # Create the save directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)

#     # Get the model name dynamically
#     model_name = model.__class__.__name__
#     plot_path = os.path.join(save_dir, f"{model_name.lower()}_training_loss_plot.png")

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for q1, q2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
#             q1, q2, labels = q1.to(device), q2.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(q1, q2)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         epoch_loss = total_loss / len(train_loader)
#         train_losses.append(epoch_loss)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

#     # Plot the training loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title(f'Training Loss for {model_name}')
#     plt.legend()
#     plt.grid(True)

#     # Save the plot
#     plt.savefig(plot_path)
#     print(f"Training loss plot saved to {plot_path}")
#     plt.close()




import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Train the model, save the trained model, and save the training loss plot.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train the model on (CPU or GPU).
        epochs: Number of training epochs.
        save_dir: Directory to save models and plots.
    """
    # Move model to device
    model.to(device)
    save_dir="models"
    # Create the base save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get model-specific directory
    model_name = model.__class__.__name__.lower()
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Paths for saving
    model_path = os.path.join(model_dir, f"{model_name}_final.pth")
    plot_path = os.path.join(model_dir, f"{model_name}_training_loss_plot.png")

    # Train the model
    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for q1, q2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            q1, q2, labels = q1.to(device), q2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(q1, q2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {model_name.capitalize()}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(plot_path)
    print(f"Training loss plot saved to {plot_path}")
    plt.close()
