# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as dataset
import torch.nn as nn
import cv2
import numpy as np
# Press the green button in the gutter to run the script.
import os
from PIL import Image
if __name__ == '__main__':
 n_mean=0.1307
 n_dev=0.3081
 baseDir='./data'
 trainDir=os.path.join(baseDir,'train')
 testDir=os.path.join(baseDir,'test')
 transforms=trans.Compose([trans.ToTensor(),trans.Normalize([n_mean],[n_dev])])
 train_dataset=dataset.MNIST(root=trainDir,transform=transforms,train=True,download=True)
 test_dataset = dataset.MNIST(root=testDir, transform=transforms, train=False, download=True)
 # print(train_dataset.data.shape)
 # print(train_dataset.targets.shape)
 image=train_dataset[20][0].numpy()*n_dev+n_mean
 print(image.shape)
 # plt.imshow(image.reshape(28,28),cmap='gray')
 # plt.show()
 bath_size_m=100
 train_load=DataLoader(dataset=train_dataset,batch_size=bath_size_m,shuffle=True)
 test_load=DataLoader(dataset=test_dataset,batch_size=bath_size_m,shuffle=False)
 # print(len(train_load))
 # print(len(test_load))
 # print(type(train_dataset))
 # print(type(train_load))
 # print(train_load.batch_size)
 # it=iter(train_load)
 # for _ in range(10) :
 #  image_batch,label_batch=next(it)
 # print(image_batch.shape)
 # print(label_batch.shape)
 # it=iter(train_dataset)
 # image2,label2=next(it)
 # print(image2.shape)
 # print(label2)
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
loss_fn=nn.CrossEntropyLoss()

# for i,(inputs,label) in enumerate(train_load):
#     if CUDA:
#        inputs=inputs.cuda()
#        label=label.cuda()
#
#     print("input size",inputs.shape)
#     print("label shape",label.shape)
#     output = model(inputs)
#     print("outputs shape",output.shape)
#     _,prediction=torch.max(output,1)
#     print("prediction shape",prediction.shape)
#     print(prediction)
#     break

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs=8
train_loss=[]
train_accuracy=[]
test_loss=[]
test_accuracy=[]
# for epocch in range(num_epochs):
#  correct=0
#  iteration=0
#  train_iter_loss_sum=0.0# every epcho the loss sum
#  model.train()
#  # Train Phase:
#  for i,(inputs,lables) in enumerate(train_load):# every batch  every batch backpropagation and upgrade
#        if CUDA:
#           inputs=inputs.cuda()
#           lables=lables.cuda()
#        outputs=model(inputs)
#        loss=loss_fn(outputs,lables)# already mean
#        train_iter_loss_sum+=loss.item()#sum loss in on epcho, every iteration everage loss
#        optimizer.zero_grad()#如果不清除会累积
#        loss.backward()#calculate backpropagation
#        optimizer.step()#upgrade
#        _,prediction=torch.max(outputs,1)# reduce dimation 1, from [100,10]->[100],
#        #first is maxvale in every row, second is maxvalue corresponding index
#        correct+=(prediction==lables).sum().item()#one epcho how many correct
#        iteration+=1
#
#  train_loss.append(train_iter_loss_sum/iteration)# loss in one epcho
#  train_accuracy.append(correct/len(train_load.dataset))#accuracy in one epcho
#
# #Testing Phase:
#  correct=0
#  test_iter_loss_sum=0
#  iteration=0
#  model.eval()
#
#  for i,(inputs,lables) in enumerate(test_load):
#       if CUDA:
#                  inputs = inputs.cuda()
#                  lables = lables.cuda()
#       outputs = model(inputs)
#       loss=loss_fn(outputs,lables)# already mean
#       test_iter_loss_sum+=loss.item()#sum loss in on epcho, every iteration everage loss
#       _, prediction = torch.max(outputs, 1)
#       correct+=(prediction==lables).sum().item()#one epcho how many correct
#       iteration+=1
#  test_loss.append(test_iter_loss_sum/iteration)
#  test_accuracy.append(correct/len(test_dataset))
#  print(f" epcho {epocch+1}/{num_epochs}  trainLoss{train_loss[epocch]:.3f}   trainAccuracy{train_accuracy[epocch]:.3f}   testLoss{test_loss[epocch]:.3f}   testAccuracy{test_accuracy[epocch]:.3f}")
#  if test_accuracy[-1]>=0.99 :
#       torch.save(model.state_dict(),f"model_epcho_{epocch}_accuracy_{test_accuracy[-1]:.3f}.pth")
# #loss show
#
# epochs = range(1, len(train_loss) + 1)  # x轴：1, 2, 3, ...
#
# plt.figure(figsize=(10, 6))  # 创建画布
#
# plt.subplot(1,2,1)
# plt.plot(epochs, train_loss, label='Train Loss', marker='o', linestyle='-')
# plt.plot(epochs, test_loss, label='Test Loss', marker='x', linestyle='--')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training vs Testing Loss")
# plt.legend()
# plt.grid(True)
#
# plt.subplot(1,2,2)
# plt.plot(epochs, train_accuracy, label='Train Accuracy', marker='o', linestyle='-')
# plt.plot(epochs, test_accuracy, label='Test Accuracy', marker='x', linestyle='--')
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training vs Testing Accuracy")
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()


# test_my_CNN

#
# test_input=train_dataset[20][0]
# test_label=train_dataset[20][1]
# test_input=test_input.view(1,1,28,28)
# test_model=CNN()
# test_model.load_state_dict(torch.load("model_epcho_4_accuracy_0.993.pth"))
#
# test_model=test_model.cuda()
# test_input=test_input.cuda()
#
# test_model.eval()
#
# test_output=test_model(test_input)
# _,out_label=torch.max(test_output,1)
# test_image=test_input.squeeze().cpu().numpy()*n_dev+n_mean
# plt.imshow(test_image, cmap='gray')
# plt.xticks([])
# plt.yticks([])
# plt.text(14, 30, f"Prediction: {out_label.item()}", ha='center', fontsize=12)
# plt.show()


# test_model=CNN()
# test_model.load_state_dict(torch.load("model_epcho_4_accuracy_0.993.pth"))
# test_model=test_model.cuda()
# test_model.eval()
#
# transform_picture=trans.Compose([trans.Resize([28,28]),trans.ToTensor(),trans.Normalize([n_mean],[n_dev])])
# image=cv2.imread("test_picture2.png", 0)
# ret, thresholded = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
# image=255-thresholded
# cv2.imshow('Original', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img = Image.fromarray(image)  # 转为 PIL 图像，方便 torchvision 处理
# img = transform_picture(img)  # 应用图像预处理（见第二张图）
# img = img.view(1, 1, 28, 28)  # 调整形状为 [batch_size=1, channel=1, height=28, width=28]
#
#
# img = img.cuda()
#
#
# output = test_model(img)
# print(output)
# print(output.data)
#
# _, predicted = torch.max(output, 1)  # 获取最大得分的类索引
# print(f" the prediction is {predicted.item()}")

# resnet practice