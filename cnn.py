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
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,16,5,1,2) # 输入1*28*28，输出16*28*28
        self.conv2=nn.Conv2d(16,32,5,1,2) # 输入16*14*14，输出32*14*14
        self.fc1=nn.Linear(32*7*7,128) # 输入32*7*7，输出128
        self.fc2=nn.Linear(128,10) #输入128，输出10
    def forward(self,x):
        conv1_out = F.max_pool2d(F.relu(self.conv1(x)),2) # 卷积池化
        conv2_out = F.max_pool2d(F.relu(self.conv2(conv1_out)),2) # 卷积池化
        res = conv2_out.view(conv2_out.size(0),-1) # 将其压缩成一维
        fc1_out = F.relu(self.fc1(res)) # 两个全连接层
        fc2_out = self.fc2(fc1_out)
        return F.log_softmax(fc2_out)

train_data = torchvision.datasets.MNIST('./mnist',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST('./mnist',train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_loader = Data.DataLoader(dataset=train_data,batch_size=50,shuffle=True)
test_loader = Data.DataLoader(dataset=test_data,batch_size=1,shuffle=True)

cnn=CNN()
cnn=cnn.cuda() # gpu运行
loss_fc=nn.CrossEntropyLoss() #损失函数
optimizer = optim.Adam(cnn.parameters(),lr=0.01) #优化器

#模型训练
def train(epoch):
    train_loss=0
    train_acc=0
    for step,(batch_x,batch_y) in enumerate(train_loader):
        batch_x=batch_x.cuda() # gpu运行
        batch_y = batch_y.cuda()
        batch_x.requires_grad=True # 对抗训练需要的batch_x的梯度
        out= cnn(batch_x)
        pred = torch.max(out,1)[1]
        train_acc+=(pred==batch_y).sum().item()

        '''
        进行对抗训练的损失函数
        loss1 = loss_fc(out,batch_y) # loss1= J(θ, x, y) 
        optimizer.zero_grad() # 梯度清零
        loss1.backward(retain_graph=True) # 梯度回传
        grad=batch_x.grad.data # 获得batch_x的梯度
        out = cnn(batch_x+0.1*grad.sign()) 
        loss2 = loss_fc(out,batch_y) # loss2 = J(θ, x + e*sign (∇xJ(θ, x, y))
        loss = 0.5*loss1 + 0.5*loss2 # loss = α*J(θ, x, y) + (1 - α)*J(θ, x + e* sign (∇xJ(θ, x, y))
        '''

        loss = loss_fc(out,batch_y) # 正常模型的损失函数
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    print('epoch {} ：Train Loss: {:.6f}, Acc: {:.6f}'.format(epoch,train_loss / (len(train_data)), train_acc / (len(train_data))))
    #if train_acc / (len(train_data)) > 0.99 : return 1
    return 0
#测试函数
def test():
    cnn.eval()
    test_loss=0
    test_acc=0
    for step,(batch_x,batch_y) in enumerate(test_loader):
        batch_x=batch_x.cuda()
        batch_y = batch_y.cuda()
        out= cnn(batch_x)
        loss = loss_fc(out,batch_y)
        test_loss+=loss.item()
        pred = torch.max(out,1)[1]
        test_acc+= (pred==batch_y).sum().item()
    print('Test Loss: {:.6f}, Accuracy: {:.6f}'.format(test_loss / (len(test_data)), test_acc/len(test_data) ))
#进行20个epoch的迭代训练
for i in range(20):
    if train(i):break
test() # 测试
torch.save(cnn.state_dict(),'mnist.pth') # 保存模型
