import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


# data setup
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


# creating our forward N.N. :

# 1o def -> creating my 4 fully connected  layers(fc1=input,fc2+fc3=hidden,fc4=output layer)
# Applies a linear transformation to the  incoming data: y = x*W^T + b , with W(weights) randomly initialized

# 2o def(forward) -> pernaei ta fc1,fc2,fc3  apo activation function(ReLU), profanos oxi to output layer
# dioti to fc4 tha perasei apo thn softmax
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)       #log(softmax)
        # return F.softmax(x, dim=1)


net = Net()
print(net)

X = torch.randn(28, 28)
X = X.view(-1, 28 * 28)


# -1 because :  This is a way of telling the library: "give me a tensor that has these many columns
# and YOU compute the appropriate number of rows that is necessary to make this happen".

print(X.shape)
output = net(X)
print(output)
