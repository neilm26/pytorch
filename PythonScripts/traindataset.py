import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from datasetcreator import WatermelonDataset

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

input_size = 3
num_classes = 2
learning_rate = 0.001
batch_size = 32
epochs = 5

path = "C:/Users/gdbio/PycharmProjects/pytorch/PythonScripts/side_of_watermelons/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((224,224), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)
resize()

side_dataset = WatermelonDataset(
    csv_file="watermelon_csv.csv",
    root_dir_imgs="side_of_watermelons",
    transform=transforms.ToTensor(),

)

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


print(side_dataset.__len__())
train_dataset, test_dataset = torch.utils.data.random_split(side_dataset, [80, 72])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = torchvision.models.MobileNetV2()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

    losses = []

    for batch_index, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)

        loss = criterion(scores, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():  # don't compute color in black and white images
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
#
model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('model.pt') # Save
optimized_scripted_module = optimize_for_mobile(model_scripted)
optimized_scripted_module._save_for_lite_interpreter("model.ptl")
