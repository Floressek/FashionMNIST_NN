import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing
import time

# This is the key addition for Windows multiprocessing
if __name__ == '__main__':
    # Optional: Only needed if you plan to create executables
    multiprocessing.freeze_support()

    print(f"Using PyTorch version: {torch.__version__}")
    print(f"Using Torchvision version: {torchvision.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if CUDA is properly set up
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    learning_rate = 0.001
    num_epochs = 20
    batch_size = 256  # Increased from 128 for better GPU utilization

    # Define transformations for the data
    # 1. ToTensor(): Converts PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    #                to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    # 2. Normalize(): Normalizes a tensor image with mean and standard deviation.
    #                 (mean,), (std,) for grayscale images. MNIST mean/std are approx known.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transform,
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transform,
                                              download=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,  # Reduce to avoid overhead
                              pin_memory=True,
                              persistent_workers=True,  # Reuse workers
                              prefetch_factor=2)  # Prefetch more batches

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True,
                             persistent_workers=True)


    class MNIST_Net(nn.Module):
        def __init__(self, input_dim=28 * 28, hidden1_dim=512, hidden2_dim=256, output_dim=10):
            super(MNIST_Net, self).__init__()
            self.flatten = nn.Flatten()  # Flattens the 28x28 image to a 784 vector
            self.layer1 = nn.Linear(input_dim, hidden1_dim)
            self.activation1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.2)
            self.layer2 = nn.Linear(hidden1_dim, hidden2_dim)
            self.activation2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.2)
            self.layer3 = nn.Linear(hidden2_dim, output_dim)

        def forward(self, x):
            # x shape: [batch_size, 1, 28, 28]
            x = self.flatten(x)  # Shape becomes: [batch_size, 784]
            x = self.layer1(x)
            x = self.activation1(x)
            x = self.dropout1(x)
            x = self.layer2(x)
            x = self.activation2(x)
            x = self.dropout2(x)
            logits = self.layer3(x)
            return logits


    model = MNIST_Net().to(device)
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # For multi-class classification, CrossEntropyLoss is standard.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Warm up GPU
    print("\nWarming up GPU...")
    with torch.no_grad():
        dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
        for _ in range(5):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("GPU warm-up complete.")

    print(f"\n--- Starting Training Loop (Epochs={num_epochs}, Batch Size={batch_size}) ---")
    n_total_steps = len(train_loader)

    # Time tracking
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        model.train()

        # --- Batch Loop (DataLoader handles batching and shuffling) ---
        for i, (images, labels) in enumerate(train_loader):
            # DataLoader provides batches of images and their corresponding labels

            images = images.to(device, non_blocking=True)  # Shape: [batch_size, 1, 28, 28]
            labels = labels.to(device, non_blocking=True)  # Shape: [batch_size] (containing class indices 0-9)

            # 8. Forward Pass: Compute predictions (logits)
            outputs = model(images)  # Shape: [batch_size, 10]

            # 9. Compute Loss: Measure the batch error
            loss = criterion(outputs, labels)

            # 10. Backward Pass & 11. Optimization Step
            # a. Zero the gradients from previous steps
            optimizer.zero_grad()
            # b. Perform backpropagation to compute gradients (dL/dW, dL/db)
            loss.backward()
            # c. Update model parameters using computed gradients
            optimizer.step()

            epoch_loss += loss.item()

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_time = time.time() - epoch_start
        average_epoch_loss = epoch_loss / n_total_steps
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {average_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\n--- Training finished in {total_time:.2f} seconds ---")
    print(f"Average time per epoch: {total_time / num_epochs:.2f} seconds")

    print("\n--- Evaluating on Test Set ---")
    model.eval()
    with torch.no_grad():  # Disable gradient calculation during evaluation
        n_correct = 0
        n_samples = 0
        test_loss = 0.0
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get predictions: The class with the highest score (logit) is the prediction
            _, predicted_labels = torch.max(outputs.data, 1)

            n_samples += labels.size(0)
            n_correct += (predicted_labels == labels).sum().item()

        average_test_loss = test_loss / len(test_loader)
        accuracy = 100.0 * n_correct / n_samples
        print(f'Average Test Loss: {average_test_loss:.4f}')
        print(f'Accuracy on Test Set: {accuracy:.2f} %')
