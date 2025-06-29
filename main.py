# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as dataset
import torch.nn as nn
# Press the green button in the gutter to run the script.
import os
if __name__ == '__main__':
 n_mean=0.1307
 n_dev=0.3081
 baseDir='./data'
 trainDir=os.path.join(baseDir,'train')
 testDir=os.path.join(baseDir,'test')
 transforms=trans.Compose([trans.ToTensor(),trans.Normalize([n_mean],[n_dev])])
 train_dataset=dataset.MNIST(root=trainDir,transform=transforms,train=True,download=True)
 test_dataset = dataset.MNIST(root=testDir, transform=transforms, train=False, download=True)
 print(train_dataset.data.shape)
 print(train_dataset.targets.shape)
 image=train_dataset[20][0].numpy()*n_dev+n_mean
 print(image.shape)
 plt.imshow(image.reshape(28,28),cmap='gray')
 plt.show()
 bath_size_m=100
 train_load=DataLoader(dataset=train_dataset,batch_size=bath_size_m,shuffle=True)
 test_load=DataLoader(dataset=test_dataset,batch_size=bath_size_m,shuffle=False)
 print(len(train_load))
 print(len(test_load))
 print(type(train_dataset))
 print(type(train_load))
 print(train_load.batch_size)
 it=iter(train_load)
 for _ in range(10) :
  image_batch,label_batch=next(it)
 print(image_batch.shape)
 print(label_batch.shape)
 it=iter(train_dataset)
 image2,label2=next(it)
 print(image2.shape)
 print(label2)
 print(torch.cuda.is_available())
 class CNN(nn.Module):
  def __init__(self):
   super(CNN,self).__init__()
   self.con1=nn.Conv2d(in_channels=1,out_channels=8,padding=1,stride=1,kernel_size=3)
   # 8 cahnnel 28*28
   self.batN1=nn.BatchNorm2d(8)
   self.relu=nn.ReLU()
   self.maxP=nn.MaxPool2d(kernel_size=2)
   #8 channel 14*14
   self.con2=nn.Conv2d(in_channels=8,out_channels=32,padding=2,stride=1,kernel_size=5)
   # 32 channel  14*14
   self.batN2 = nn.BatchNorm2d(32)
   # maxPlool 32 channel 7*7
   #fc1
   self.fc1=nn.Linear(1568,600)
   self.dropout=nn.Dropout(p=0.5)
   self.fc2=nn.Linear(600,10)
  def forward(self,x):
   batch_size = x.size(0)
   out=self.con1(x)
   out=self.batN1(out)
   out=self.relu(out)
   out=self.maxP(out)
   out=self.con2(out)
   out=self.batN2(out)
   out=self.relu(out)
   out=self.maxP(out)
   out=out.view(batch_size,-1)
   out=self.fc1(out)
   out=self.relu(out)
   out=self.dropout(out)
   out=self.fc2(out)
   return out

model=CNN()
CUDA=torch.cuda.is_available()
if CUDA:
 model=model.cuda()
for i,(inputs,label) in enumerate(train_load):
 if CUDA:
  inputs=inputs.cuda()
  label=label.cuda()




