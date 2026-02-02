#该部分为训练模型
#保存数据，设置参数，使用网络，训练，验证，保存模型
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from modelll import *
from torch.utils.data import DataLoader
import os

#读取数据集
transform=transforms.Compose([
    transforms.Resize((32, 32)),  #调整尺寸为 32×32
    transforms.ToTensor(),  #PIL转张量，像素值缩放到[0, 1]，形状变为 (3, 32, 32)
    transforms.Normalize((0.5,), (0.5,)),  # 步骤4：3 通道标准化（配套修改）
])

train_data = torchvision.datasets.ImageFolder(os.path.join("traindir"),transform=transform)
test_data = torchvision.datasets.ImageFolder(os.path.join("testdir"),transform=transform)
test_data_size=len(test_data)

#读取数据集
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=True)

#实例化网络模型
class Discern(nn.Module):
    def __init__(self):
        super(Discern,self).__init__()
        self.model=torch.nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(in_features=64,out_features=4)
        )
    def forward(self,x):
        x=self.model(x)
        return x
fs=Discern()
fs=fs.cuda()

#损失函数
loss_func=nn.CrossEntropyLoss()
loss_func=loss_func.cuda()
#优化器
optimizer=torch.optim.SGD(fs.parameters(),lr=0.001,momentum=0.9)

writer=SummaryWriter("log")

#设置训练参数
epochs=100 #10次
step=0
for i in range(epochs):
    for data in train_loader:
        img,label=data #图片和标签
        img=img.cuda()
        label=label.cuda()
        output=fs(img)
        loss=loss_func(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step=step + 1
        #print(step,loss.item())

    fs.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_loader:
            img,label=data
            img=img.cuda()
            label=label.cuda()
            output=fs(img)
            loss=loss_func(output,label)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(output.argmax(1)==label).sum()
            total_accuracy=total_accuracy+accuracy.item()

    #print("Loss{}",format(total_test_loss,".2f"))
    #print("Accuracy{}",format(total_accuracy/test_data_size,".2f"))
    writer.add_scalar("loss",total_test_loss,step)
    writer.add_scalar("accuracy",total_accuracy/test_data_size,step)
    step+=1

    torch.save(fs.state_dict(),"FS_model_comp.pth")
    print("已保存")

writer.close()