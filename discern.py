#使用模型进行识别
import torchvision
from PIL import Image
import torch
from torch import nn

img_path="org.jpg"
image=Image.open(img_path)

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])

image=transform(image)

class Discern(nn.Module): #还原完整模型，模型结构
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
            nn.Linear(in_features=64,out_features=4),  #四分类，out=4
        )

    def forward(self,x):
        x=self.model(x)
        return x

fs=Discern() #实例化
model=torch.load("FS_model_comp.pth") #加载预训练权重，模型权重
#print(model)
fs.load_state_dict(model) #将权重填充到神经网络中，使结构完整

image=torch.reshape(image,(1,3,32,32))
fs.eval() #切换为测试模式
with torch.no_grad(): #关闭梯度计算，节省内存和计算资源
    output=fs(image)

print(output)
print(output.argmax(dim=1))