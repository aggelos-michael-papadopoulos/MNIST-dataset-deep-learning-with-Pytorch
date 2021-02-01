import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# data setup
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


net = ConvNet()

# declaring loss function and my optimizer (gradient descent algorithm -> for the weight updating)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=.001)

# training
for epoch in range(3):                      # 3 full passes over the data (3 epochs)
    for data in trainset:                   # `data` is a batch of data (1 batch which has 10 28x28x1 images)
        X, y = data                         # X is the batch of features(10,1,28,28), y is the batch of targets(labels).
        net.zero_grad()                     # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.view(-1, 784))       # pass in the reshaped batch:10 fores flatten gia kathe 28x28x1 image apo to batch
        loss = F.nll_loss(output, y)        # calc and grab the loss value
        loss.backward()                     # apply this loss backwards thru the network's parameters
        optimizer.step()                    # attempt to optimize weights to account for loss/gradients
    print(loss)                             # print loss. We hope loss (a measure of wrong-ness) declines!


# testing
correct = 0
total = 0
with torch.no_grad():                       # deactivates autograd engine
    for data in testset:
        X, y = data
        output = net(X.view(-1, 784))       # flattening.. einai 1000 1x10  tesnors afoy exoyme 10 sto softmax(10000 test data)
        # print(output)
        for idx, i in enumerate(output):
            # print(torch.argmax(i), y[idx])  # print arxika thn problepsi(NN gia to test label, kai meta deixnei raw?test label
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1


print('Accuracy: ', round(correct/total, 3)*100, '%')

# vizualize
print('the test set is: ')
plt.imshow(X.view(280, 28))                         # the 10 label testset kai emeis kanoume check to 6o me to NN
plt.show()
print('the testing image is: ')
plt.imshow(X[5].view(28, 28))
plt.show()
print('and the prediction from the NN for this image is the number: '
      , int(torch.argmax(net(X[5].view(-1, 784))[0])))           # το 5ο output-προβλεψη tou output_τεστ του ΝΝ για την 5η εικονα απο το batch(twn 10)

