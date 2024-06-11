---

# CNN Training: CPU vs GPU with PyTorch

This project demonstrates the performance difference between training a Convolutional Neural Network (CNN) on a CPU versus a GPU using PyTorch. We use the CIFAR-10 dataset, a popular dataset for image classification tasks.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The aim of this project is to compare the training time of a simple CNN on CPU and GPU. We leverage PyTorch for model definition, training, and evaluation. The CIFAR-10 dataset is used to train the model.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.x
- PyTorch
- torchvision
- matplotlib

If you have a GPU and want to utilize CUDA, ensure that CUDA and cuDNN are properly installed.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/cnn-cpu-vs-gpu.git
    cd cnn-cpu-vs-gpu
    ```

2. Install the required Python packages:
    ```bash
    pip install torch torchvision matplotlib
    ```

## Usage

To run the training and compare the performance:

1. Ensure you are in the project directory.
2. Execute the script:
    ```bash
    python main.py
    ```

The script will train the CNN on both the CPU and GPU (if available) and display the training times and speedup.

## Results

The script outputs the training time for both CPU and GPU, calculates the speedup, and plots a comparison graph. Here’s an example of the expected output:

```
Training on CPU...
[Epoch: 1, Batch: 100] loss: 2.303
...
Training completed in 120.45 seconds

Training on GPU...
[Epoch: 1, Batch: 100] loss: 2.303
...
Training completed in 35.78 seconds

Training time on CPU: 120.45 seconds
Training time on GPU: 35.78 seconds
Speedup: 3.37x
GPU is 70.31% faster than CPU
```

## Code Explanation

### Loading the Dataset

We use the CIFAR-10 dataset, which is loaded and transformed using the following code:

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)
```

### Defining the CNN Model

The CNN model is defined as follows:

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Training the Model

The training function handles the training loop and time measurement:

```python
import torch.optim as optim
import time

def train_model(device, epochs=2):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    return training_time
```

### Comparing Performance

After training, we compare the performance:

```python
if __name__ == '__main__':
    # Training on CPU
    device = torch.device("cpu")
    print("Training on CPU...")
    cpu_time = train_model(device)

    # Training on GPU (if available)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Training on GPU...")
        gpu_time = train_model(device)
    else:
        print("CUDA is not available. GPU training is skipped.")
        gpu_time = None

    if gpu_time is not None:
        speedup = cpu_time / gpu_time
        percentage_faster = (cpu_time - gpu_time) / cpu_time * 100
        print(f'Training time on CPU: {cpu_time:.2f} seconds')
        print(f'Training time on GPU: {gpu_time:.2f} seconds')
        print(f'Speedup: {speedup:.2f}x')
        print(f'GPU is {percentage_faster:.2f}% faster than CPU')
        
        # Plotting the graph
        import matplotlib.pyplot as plt

        devices = ['CPU', 'GPU']
        times = [cpu_time, gpu_time]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(devices, times, color=['blue', 'orange'])
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison: CPU vs GPU')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate the bars with the actual training times
        for bar, time in zip(bars, times):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} s', ha='center', va='bottom')

        # Add percentage improvement
        plt.text(0.5, max(times) * 0.95, f'GPU is {percentage_faster:.2f}% faster than CPU',
                 ha='center', va='top', fontsize=12, color='green')

        plt.show()
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
