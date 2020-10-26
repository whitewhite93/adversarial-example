import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data.dataloader as Data
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)  # 输入1*28*28，输出16*28*28
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)  # 输入16*14*14，输出32*14*14
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 输入32*7*7，输出128
        self.fc2 = nn.Linear(128, 10)  # 输入128，输出10

    def forward(self, x):
        conv1_out = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 卷积池化
        conv2_out = F.max_pool2d(F.relu(self.conv2(conv1_out)), 2)  # 卷积池化
        res = conv2_out.view(conv2_out.size(0), -1)  # 将其压缩成一维
        fc1_out = F.relu(self.fc1(res))  # 两个全连接层
        fc2_out = self.fc2(fc1_out)
        return F.log_softmax(fc2_out)

train_data = torchvision.datasets.MNIST('./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST('./mnist', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)

cnn = CNN()
cnn.load_state_dict(torch.load('mnist.pth'))  # 加载模型参数
cnn = cnn.cuda()  # gpu运行
loss_fc = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(cnn.parameters(), lr=0.01)  # 优化器
epsilon = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def fgsm_attack(image, grad, e):
    sign = grad.sign()  # 取梯度的方向
    per_image = image + e * sign  # 核心代码
    pre_image = torch.clamp(per_image, 0, 1)  # 把修改后的image都转到(0,1)区间内
    return per_image

# 进行攻击
def test(i):
    corr = 0
    adv_ex = []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.cuda()  # 转到gpu运行
        batch_y = batch_y.cuda()
        batch_x.requires_grad = True  # 使batch_x能够梯度回传
        out = cnn(batch_x)
        pred_1 = torch.max(out, 1)[1]  # 获得概率最大的索引
        if pred_1.item() != batch_y.item(): continue  # 如果已经和标签不同，则不需要攻击
        loss = loss_fc(out, batch_y)  # 损失函数
        optimizer.zero_grad()  # 梯度清空
        loss.backward()  # 梯度回传
        grad = batch_x.grad.data  # 得到batch_x的梯度
        per_batch_x = fgsm_attack(batch_x, grad, epsilon[i])  # 进行攻击
        out = cnn(per_batch_x)
        pred_2 = torch.max(out, 1)[1]
        if pred_1.item() != pred_2.item():  # 攻击成功
            if len(adv_ex) < 5:  # 存放5张图片
                ex = per_batch_x.squeeze().detach().cpu().numpy()  # 将tensor变量转为numpy
                adv_ex.append((pred_1.item(), pred_2.item(), ex))  # 图片
        else:
            corr += 1  # 攻击失败
            if (i == 0) and (len(adv_ex) < 5):  # 存放没有攻击时的图片
                ex = per_batch_x.squeeze().detach().cpu().numpy()
                adv_ex.append((pred_1.item(), pred_2.item(), ex))

    print('Epsilon:{}   Test Accuracy:{}'.format(epsilon[i], corr / len(test_loader)))
    return corr / len(test_loader), adv_ex


# 进行不同epsilon取值的攻击
cnt = 0
accuracy = []
example = []
for i in range(len(epsilon)):
    acc, ex = test(i)
    accuracy.append(acc)
    example.append(ex)

# 画图
plt.figure(figsize=(8, 10))
for i in range(len(epsilon)):
    for j in range(len(example[i])):
        cnt += 1
        plt.subplot(len(epsilon), len(example[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilon[i]))
        a, b, c = example[i][j]
        plt.title('{}->{}'.format(a, b))
        plt.imshow(c)
plt.tight_layout()
plt.show()
plt.figure(figsize=(5, 5))
plt.plot(epsilon, accuracy, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.35, step=0.05))
plt.xlabel("epsilon")
plt.ylabel("accuracy")
plt.show()



