import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

train_dataset_pil = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               download=True)

transform = transforms.ToTensor()
train_dataset_tensor = torchvision.datasets.MNIST(root='./data',
                                                  train=True,
                                                  transform=transform,
                                                  download=True)

sample_index = 0
image_pil, label_pil = train_dataset_pil[sample_index]
image_tensor, label_tensor = train_dataset_tensor[sample_index]

print("--- Original PIL Image ---")
print(f"Image Type: {type(image_pil)}")
print(f"Image Size (Width, Height): {image_pil.size}") # PIL uses (width, height)
print(f"Image Mode (e.g., 'L' for grayscale): {image_pil.mode}")
print(f"Label: {label_pil}")
print(f"Label Type: {type(label_pil)}")

print("\n--- Transformed PyTorch Tensor ---")
print(f"Tensor Type: {type(image_tensor)}")
print(f"Tensor Shape (Channels, Height, Width): {image_tensor.shape}")
print(f"Tensor Data Type: {image_tensor.dtype}")
print(f"Tensor Min Value: {image_tensor.min():.4f}") # ToTensor scales to [0.0, 1.0]
print(f"Tensor Max Value: {image_tensor.max():.4f}")
print(f"Label: {label_tensor}")
print(f"Label Type: {type(label_tensor)}")

plt.figure(figsize=(4, 4))
plt.imshow(image_pil, cmap='gray')
plt.title(f"MNIST Sample #{sample_index} - Label: {label_pil}")
plt.axis('off')
plt.show()