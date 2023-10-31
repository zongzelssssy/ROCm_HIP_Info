**BY DESIGN. YOU SHOULD NOT NEED TO PORT YOUR PYTORCH WORKLOADS FOR ROCM**



• ROCm PyTorch has a similar interface to CUDA PyTorch

• torch.cuda module works the same for HIP

• cuda device is the HIP device: model.to (device="cuda")

• Full feature parity with CUDA PyTorch

• CUDAExtension runs hipify for you automatically

•If you really want to know if you're CUDA or HIP, `is_rom_pytorch = torch.version.hip` is not None



So that you can dierctly use ROCm_pytorch as same as CUDA_pytorch.



Run a simple demo of **Cigar-100 ResNet** :

First access the GPU node `srun -p mi100 --pty bash -i`

and download the demo pythob file : pytorch_train.py

Then run `python pytorch_train.py` directly.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load and pre-process data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR100(root='***/cifar-100', train=True,
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='***/cifar-100', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

# Define ResNet model
net = torchvision.models.resnet18(pretrained=False)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 100)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Train 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

# Test
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Assess which troch you used
print(torch.version.hip)
print(torch.version.cuda)
```



You can get this similar result:

```Epoch [1/10], Step [100/391], Loss: 4.3227
Epoch [1/10], Step [200/391], Loss: 3.8862
Epoch [1/10], Step [300/391], Loss: 3.6923
Epoch [2/10], Step [100/391], Loss: 3.3880
Epoch [2/10], Step [200/391], Loss: 3.3118
Epoch [2/10], Step [300/391], Loss: 3.2547
Epoch [3/10], Step [100/391], Loss: 3.0523
Epoch [3/10], Step [200/391], Loss: 3.0070
Epoch [3/10], Step [300/391], Loss: 3.0092
Epoch [4/10], Step [100/391], Loss: 2.8245
Epoch [4/10], Step [200/391], Loss: 2.8455
Epoch [4/10], Step [300/391], Loss: 2.8121
Epoch [5/10], Step [100/391], Loss: 2.6620
Epoch [5/10], Step [200/391], Loss: 2.6469
Epoch [5/10], Step [300/391], Loss: 2.6376
Epoch [6/10], Step [100/391], Loss: 2.5107
Epoch [6/10], Step [200/391], Loss: 2.4573
Epoch [6/10], Step [300/391], Loss: 2.4763
Epoch [7/10], Step [100/391], Loss: 2.3136
Epoch [7/10], Step [200/391], Loss: 2.3718
Epoch [7/10], Step [300/391], Loss: 2.3370
Epoch [8/10], Step [100/391], Loss: 2.2421
Epoch [8/10], Step [200/391], Loss: 2.2374
Epoch [8/10], Step [300/391], Loss: 2.2727
Epoch [9/10], Step [100/391], Loss: 2.1412
Epoch [9/10], Step [200/391], Loss: 2.1437
Epoch [9/10], Step [300/391], Loss: 2.1518
Epoch [10/10], Step [100/391], Loss: 2.0446
Epoch [10/10], Step [200/391], Loss: 2.0720
Epoch [10/10], Step [300/391], Loss: 2.0675
Test Accuracy: 41.51%
5.4.22803-474e8620
None 
```



You can see last two lines which torch.version.hip show the version of it but torch.version.cuda is none.
