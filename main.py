

#%%
import scipy.io as spio
import torch
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset

import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis

dtype=torch.cuda.FloatTensor


class data_train(Dataset):
    
    def __init__(self):
        
        mat=hdf5storage.loadmat('data/flat/X_train.mat')
        x=mat['X_train']
        mat=spio.loadmat('data/flat/y_train.mat', squeeze_me=True)
        y=mat['y_train']
        mat=spio.loadmat('data/flat/z_train.mat')
        z=mat['z_train']
        mat=spio.loadmat('data/flat/m_train.mat', squeeze_me=True)
        m=mat['m_train']
        
        # x is a 15x15x4 patch.
        x=np.transpose(x,(3,2,0,1))
        self.x=torch.FloatTensor(x.astype(float))
        
        # y is an output pixel.
        y=y[:,newaxis,newaxis,newaxis]
        self.y=torch.FloatTensor(y.astype(float))
        
        # z is some kind of prior information which is depreciated.
        # We don't use z in the paper version of the network, though is still in the dataset.
        z=np.transpose(z,(1,0))
        z=z[:,1,newaxis,newaxis]
        self.z=torch.FloatTensor(z.astype(float))
        
        # m is a mask label, which helps the reshaping and visualization of the output image.
        m=m[:,newaxis,newaxis,newaxis]
        self.m=torch.FloatTensor(m.astype(float))
        
    def __len__(self):
        print('train:')
        print(len(self.x))
        return len(self.x)
    
    def __getitem__(self,idx):
        
        x=self.x[idx,:,:,:]
        y=self.y[idx]
        z=self.z[idx]
        m=self.m[idx]
        
        return x, y, z, m



class data_test_flat(Dataset):
       
    def __init__(self):
        
        mat=hdf5storage.loadmat('data/flat/X_test.mat')
        x=mat['X_test']
        mat=spio.loadmat('data/flat/y_test.mat', squeeze_me=True)
        y=mat['y_test']
        mat=spio.loadmat('data/flat/z_test.mat')
        z=mat['z_test']
        mat=spio.loadmat('data/flat/m_test.mat', squeeze_me=True)
        m=mat['m_test']
        
        x=np.transpose(x,(3,2,0,1))
        self.x=torch.FloatTensor(x.astype(float))
        
        y=y[:,newaxis,newaxis,newaxis]
        self.y=torch.FloatTensor(y.astype(float))
        
        z=np.transpose(z,(1,0))
        z=z[:,1,newaxis,newaxis]
        self.z=torch.FloatTensor(z.astype(float))
        
        m=m[:,newaxis,newaxis,newaxis]
        self.m=torch.FloatTensor(m.astype(float))
        
    def __len__(self):
        print('test:')
        print(len(self.x))
        return len(self.x)
    
    def __getitem__(self,idx):
        
        x=self.x[idx,:,:,:]
        y=self.y[idx]
        z=self.z[idx]
        m=self.m[idx]
#        sample={'in1':x1,'in2':x2,'in3':x3,'in4':x4,'y_train':y_train}
        
        return x, y, z, m



imsize=248
batch_size=248



#trainset = data_utils.TensorDataset(torch_X_train, torch_y_train)

trainset = data_train()
trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True)

'''
X_test=np.transpose(X_test,(3,2,0,1))
torch_X_test=torch.FloatTensor(X_test.astype(float))
y_test=y_test[:,newaxis,newaxis,newaxis]
torch_y_test=torch.FloatTensor(y_test.astype(float))
testset = data_utils.TensorDataset(torch_X_test, torch_y_test)
testloader = data_utils.DataLoader(testset, batch_size=batch_size, shuffle=False)
'''

testset = data_test_flat()
testloader = data_utils.DataLoader(testset, batch_size=batch_size, shuffle=False)

print('data loaded.')


########################################################################
# Define a Convolution Neural Network

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math



# GaussianNoise module from :
# https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev=stddev
        
    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=padding, bias=False)

# Residual Block 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv1nopad = conv3x3(in_channels, out_channels, stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        if self.downsample:
            out = self.conv1nopad(x)
        else:
            out = self.conv1(x)
#        out = self.bn1(out) #         ##############  
        out = self.relu(out)
        out = self.conv2(out)
#        out = self.bn2(out) #         ##############
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, block, planes, layers):
        self.in_channels = planes
        super(Net, self).__init__()
        
        
        self.preres = nn.Sequential(
            GaussianNoise(0.01),
            nn.Conv2d(4, planes, kernel_size=3, padding=1),
        )
        self.postres = nn.Sequential(
            nn.Conv2d(planes*4,planes*4,3),
            nn.Conv2d(planes*4,1,1),
        )
        
        self.layer1 = self.make_layer(block, planes, layers[0])
        self.layer2 = self.make_layer(block, planes*2, layers[1],2)
        self.layer3 = self.make_layer(block, planes*4, layers[2],2)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride, padding=0),
#                nn.BatchNorm2d(out_channels), #         ##############
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.preres(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.postres(x)
        return x



net = Net(ResidualBlock, 16, [8,8,8])
print('net defined.')

net.cuda()
print('gpu enabled.')


########################################################################
# Define a Loss function and optimizer

import torch.optim as optim
from torch.optim import lr_scheduler

criterion = nn.SmoothL1Loss()
optimizer = optim.RMSprop(net.parameters(), lr=0.0003, momentum=0.5)

print('optimizer set.')

########################################################################
# Train the network
print('Strting to train.')
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,30], gamma=1.0/3)

for epoch in range(50):  # loop over the dataset multiple times
    
    scheduler.step()
    print('new_lr_is_%f' % optimizer.param_groups[0]['lr'])
    
    running_loss = 0.0
    for i, data in enumerate(trainloader): ## erased ,0
        # get the inputs
        in1, labels, z, m = data

        # wrap them in Variable
        in1 = Variable(in1.cuda())
        #z = Variable(z.cuda())
        labels = Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(in1)
        
        outputs = torch.squeeze(outputs)
        labels = torch.squeeze(labels)
        loss_net = criterion(outputs, labels)
        #loss_p2 = criterion(outputs,z)
        loss=loss_net#+0.1*loss_p2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 5000 == 4999:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

torch.save(net, 'net_0618_50iter_rmsprop_momentum8c.tar')
print('Finished Training')
'''
model=torch.load('models/net_0618_50iter_rmsprop_momentum8c.tar')
net=model
'''

########################################################################

imcount=8

correct = 0
total = 0
total_masked = 0
correct_masked = 0
outdepth = np.zeros(imsize*imsize*imcount)
truedepth = np.zeros(imsize*imsize*imcount)
masks = np.zeros(imsize*imsize*imcount)
for data in testloader:
#    images, labels = data
#    outputs = net(Variable(images))
    
    in1, labels, z, m = data

    # wrap them in Variable
    in1 = Variable(in1.cuda())
    labels = Variable(labels.cuda())
    outputs = net(in1.cuda())
    
    #_, predicted = torch.max(outputs.data, 1)
    #print(labels.size(0))
    total += labels.size(0)
    total_masked += np.sum(m.numpy().squeeze())
    difmat = (torch.squeeze(outputs.cpu())).data.numpy() - (torch.squeeze(labels.cpu())).data.numpy()
    correct += np.sum(difmat**2)
    correct_masked += np.sum(np.multiply(m.numpy().squeeze(), difmat)**2)
    outdepth[total-labels.size(0) : total] = (torch.squeeze(outputs.cpu())).data.numpy()
    truedepth[total-labels.size(0) : total] = (torch.squeeze(labels.cpu())).data.numpy()
    masks[total-labels.size(0) : total] = (torch.squeeze(m.cpu())).numpy()
    
print('MSE of test datas(this function is outdated btw): %f' % (
    (correct_masked / total_masked)**(0.5) ))



########################################################################
import scipy.misc
from PIL import Image


def trimshowsave(b, low_thresh, fname, imsize):
    high_thresh = low_thresh+255
    lv = b<low_thresh
    b[lv] = low_thresh
    hv = b>high_thresh
    b[hv] = high_thresh
    
    # https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
    bb = np.empty((imsize,imsize),dtype = np.uint8)
    bb = (b-low_thresh).astype(np.uint8)
    bb = bb.transpose()
    img=Image.fromarray(bb)
    img.show()
    scipy.misc.imsave(fname, bb)


outs = np.zeros((imsize,imsize,imcount))
trus = np.zeros((imsize,imsize,imcount))
ms = np.zeros((imsize,imsize,imcount))
# Reshape and visualize image.
for i in range(0,imcount):
    
    outdepthcut=outdepth[i*imsize*imsize : (i+1)*imsize*imsize]
    a1 = np.reshape(outdepthcut,(imsize,imsize))
    b1 = np.copy(a1)
    outs[:,:,i] = b1
    truedepthcut=truedepth[i*imsize*imsize : (i+1)*imsize*imsize]
    a2 = np.reshape(truedepthcut,(imsize,imsize))
    b2 = np.copy(a2)
    trus[:,:,i] = b2
    
    ms2=masks[i*imsize*imsize : (i+1)*imsize*imsize]
    ms1=np.reshape(ms2,(imsize,imsize))
    ms[:,:,i]=ms1
    
    trimshowsave(b1, 800, "%d_out.png" % (i),imsize)
    trimshowsave(b2, 800, "%d_truth.png" % (i),imsize)
    trimshowsave(b1-b2, -127, "%d_zdiff.png" % (i),imsize)



########################################################################
    
def maskedRMSE(out,tru,mask):
    m = np.invert((1-mask).astype(bool)).astype(float)
    mrmse = np.sqrt( np.sum( np.power( np.multiply(out-tru, m), 2 )) / (np.sum(m)) )
    return mrmse
    
    
    
# median filter loop
from scipy import ndimage
import scipy.io

outs_med = np.zeros((imsize,imsize,imcount))
orig_mse = 0
medd_mse = 0
for i in range(0,imcount):
    outs_med[:,:,i]=ndimage.median_filter(outs[:,:,i],3)
    orig_rmse = orig_mse+maskedRMSE(outs[:,:,i],trus[:,:,i],ms[:,:,i])
    medd_rmse = medd_mse+maskedRMSE(outs_med[:,:,i],trus[:,:,i],ms[:,:,i])
    print ('%f becomes %f ' % (maskedRMSE(outs[:,:,i],trus[:,:,i],ms[:,:,i]), maskedRMSE(outs_med[:,:,i],trus[:,:,i],ms[:,:,i])))
    
    im=outs_med[:,:,i]
    scipy.io.savemat('im%d'%(i),mdict={'im':im})
    #trimshowsave(outs_med[:,:,i], 700, "%d_out_med.png" % (i),imsize)

    
print('original masked rmse : %f' % orig_rmse)
print('original masked rmse : %f' % medd_rmse)

########################################################################
'''
class_sqerr = 0 ##list(0. for i in range(10))
class_total = 0 ##list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    #_, predicted = torch.max(outputs.data, 1)
    #c = (predicted.cpu() == labels).squeeze()
    c = c = np.sum(((torch.squeeze(outputs)).data.numpy() - (torch.squeeze(labels)).numpy())**2)
    for i in range(4):
        label = labels[i]
        class_sqerr += c
        class_total += 1
'''
'''
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
'''