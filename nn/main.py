import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch

batch_size = 4

# CIFAR10 类别
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

# 构建数据规范化变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 构建 torchvision.datasets.CIFAR10 对象
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 构建 torch.utils.data.DataLoader 对象
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# 模仿训练数据集对象实例的建立方式，构建测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# 可视化部分训练数据
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 随机取一个批次的训练数据
dataiter = iter(trainloader)
images, labels = next(dataiter)
# 展示图像
imshow(torchvision.utils.make_grid(images))
# 打印相关图像的标签
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = MyNet()
print("\nNetwork architecture:\n")
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

from tqdm.auto import tqdm
import time

n_epochs = 2
n_batches = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

print("Starting training...")
start_time = time.time()

for epoch in range(n_epochs):
    # 当前 n_batches 个小批次训练数据上的平均损失
    running_loss = 0.0
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{n_epochs}")
    for i, data in enumerate(pbar, 0):
        # 获取数据及标签，并转移到 GPU（若可用）
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # 梯度清零，防止累加
        optimizer.zero_grad()
        # 前向传播
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()

        # 统计损失函数的滑动平均
        running_loss += loss.item()
        if (i+1) % n_batches == 0:
            pbar.set_postfix(batch=i+1, loss=running_loss/n_batches)
            running_loss = 0.0
        pbar.update() # 更新进度条

end_time = time.time()
print("Training completed. Total time: {:.2f}s".format(end_time - start_time))

model_path = "./models/cifar_net.pth"
torch.save(net.state_dict(), model_path)
print("Model saved to", model_path)

net = MyNet() 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)  

net.load_state_dict(
    torch.load(model_path, 
             map_location=device,  # 确保权重映射到当前设备
             weights_only=True     # 消除安全警告
    )
)

correct = 0
total = 0

with torch.no_grad():# 禁用梯度计算以节省内存
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device) 
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Overall accuracy on test set: %.1f%%" % (100 * correct / total))

print("Calculating per-class accuracy...")
class_correct = [0.] * 10
class_total = [0.] * 10
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            class_correct[label] += (predicted[i] == labels[i]).item()

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))