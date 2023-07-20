import torch
from torch import nn, optim
from torchsummary import summary
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

#Load the MNIST dataset
batch_size_train = 64
batch_size_test = 64

train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

# Create a sequential model
model = nn.Sequential()

# Add 3 convolutional and pooling layers
model.add_module('Conv_1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3)))
model.add_module('Relu_1', nn.ReLU())
model.add_module('MaxPool_1', nn.MaxPool2d(kernel_size=2,stride=2))

model.add_module('Conv_2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)))
model.add_module('Relu_2', nn.ReLU())
model.add_module('MaxPool_2', nn.MaxPool2d(kernel_size=2,stride=2))

model.add_module('Conv_3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3)))
model.add_module('Relu_3', nn.ReLU())

# Add a Flatten layer to the model
model.add_module('Flatten', nn.Flatten())
# Add a Linear layer with 64 units and relu activation
model.add_module('Linear_1', nn.Linear(in_features=64*3*3, out_features=64, bias=True))
model.add_module('Relu_L_1', nn.ReLU())
# Add the last Linear layer.
model.add_module('Linear_2', nn.Linear(in_features=64, out_features=10, bias=True))
model.add_module('Out_activation', nn.Softmax(-1))

in_shape = (1,28,28)
model = model.to(device)
summary(model, input_size=(in_shape))


optimizer = optim.RMSprop(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss(reduction='mean')
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            test_loss += loss.item()
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss / len(train_loader):.4f} - Validation Loss: {test_loss / len(test_loader):.4f}")

model.eval()
test_acc = 0
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    predictions = model(images)
    accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == labels).sum() / len(labels)
    test_acc += accuracy.item()
test_acc /= len(test_loader)
print(f"Test accuracy: {test_acc:.3f}")