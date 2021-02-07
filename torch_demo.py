import matplotlib.pyplot
import numpy
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torchvision
import torchvision.transforms


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the tranning set and test set of torchvision
transform_function = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_function)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_function)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

# Instantiate the network
net = Network()

#  Define a Loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
for t in range(2):

    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        # Get the inputs
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = net(inputs)

        # Backward
        loss = loss_function(outputs, labels)
        loss.backward()

        # Optimization
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        if step % 300 == 299:  # print every 300 mini-batches
            print('Time: %d, Step: %5d loss: %.3f' %
                  (t + 1, step + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Save the state.
#  save_path = './train.pth'
#  torch.save(net.state_dict(), save_path)
#  net.load_state_dict(torch.load(save_path))

# Single test.
#  images, _ = iter(testloader).next()
#  outputs = net(images)
#
# Get the index of the highest energy
#  _, predicted = torch.max(outputs, 1)
#
#  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#  for j in range(4)))

# Test the network with the test dataset.
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))

# Detailed accuracy of each tag.
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for step in range(4):
            label = labels[step]
            class_correct[label] += c[step].item()
            class_total[label] += 1

for step in range(10):
    print('Accuracy of %5s : %2d %%' %
          (classes[step], 100 * class_correct[step] / class_total[step]))
