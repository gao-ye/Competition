from torch import nn
import torch as t
from torch.nn import functional as F
 
class ResidualBlock(nn.Module):
    #实现子module: Residual    Block
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        
        self.right=shortcut
        
    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)
    
    
class ResNet(nn.Module):
    #实现主module:ResNet34
    #ResNet34包含多个layer,每个layer又包含多个residual block
    #用子module实现residual block , 用 _make_layer 函数实现layer
    def __init__(self,num_classes= 4 ):
        super(ResNet,self).__init__()
        self.pre=nn.Sequential(
            #nn.Conv2d(3,64,7,1,3,bias=False),  #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
            nn.Conv2d(8,64,3,1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            
            # ,nn.MaxPool2d(3,1,1)
        )
        #重复的layer,分别有3,4,6,3个residual block
        self.layer1=self._make_layer(64,64,3)
        self.layer2=self._make_layer(64,128,4,stride=2)
        self.layer3=self._make_layer(128,256,6,stride=1)
        self.layer4=self._make_layer(256,512,3,stride=1)
        
        #分类用的全连接
        self.fc=nn.Linear(2048,num_classes)
        self.softmax = nn.Softmax(dim =1)
        
    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        #构建layer,包含多个residual block
        shortcut=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel))
 
        layers=[ ]
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.pre(x)
        # print("after pre is {}".format(x.shape))
        
        x=self.layer1(x)
        # print("after layer1 is {}".format(x.shape))
        x=self.layer2(x)
        # print("after layer2 is {}".format(x.shape))
        x=self.layer3(x)
        # print("after layer3 is {}".format(x.shape))
        x=self.layer4(x)
        # print("after layer4 is {}".format(x.shape))
        
        # x=F.avg_pool2d(x,3)
        #print("after avg_pool2d is {}".format(x.shape))
        x=x.view(x.size(0),-1)
        # print("after x.view is {}".format(x.shape))
        x = self.fc(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    model=ResNet()
    # input=t.autograd.Variable(t.randn(1,3,224,224))
    input=t.autograd.Variable(t.randn(1,8,4,4))
    o=model(input)
    print(o)

