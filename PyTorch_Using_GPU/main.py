import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import matplotlib.pyplot as plt

# Load CIFAR10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)


# Function to train the model
def train_model(device, epochs=10):
    model = models.resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()

    for epoch in range(epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    return training_time


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
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{yval:.2f} s', ha='center', va='bottom')

        # Add percentage improvement
        plt.text(0.5, max(times) * 0.95, f'GPU is {percentage_faster:.2f}% faster than CPU',
                 ha='center', va='top', fontsize=12, color='green')

        plt.show()
