import torch.nn as nn
import torch
import os
import cv2
import pandas as pd
import numpy as np

class ConvSpaceMask(nn.Module):
    """
    Convolution spatial mask.
    Parameters
    ----------
    n_frame: int
        Number of input frames
    n_channel: int
        Number of input channels.
    kernel_size: int
        Size of the convolutional kernel.
    bias: bool
        Whether or not to add the bias.
    """    
    def __init__(self, n_channel=3, kernel_size=3, bias=True):
        super(ConvSpaceMask, self).__init__()

        self.n_channel = n_channel
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

#        self.conv_reduction = nn.Sequential(
#                                            nn.Conv2d(in_channels=self.n_channel,
#                                                      out_channels=2,
#                                                      kernel_size=self.kernel_size,
#                                                      padding=self.padding,
#                                                      bias=self.bias),
#                                            nn.LeakyReLU(),
#                                            nn.Conv2d(in_channels=2,
#                                                      out_channels=1,
#                                                      kernel_size=self.kernel_size,
#                                                      padding=self.padding,
#                                                      bias=self.bias),
#                                            nn.LeakyReLU(),
#                                            nn.Conv2d(in_channels=1,
#                                                      out_channels=1,
#                                                      kernel_size=self.kernel_size,
#                                                      padding=self.padding,
#                                                      bias=self.bias))  
        self.conv_reduction = nn.Sequential(
                                            nn.Conv2d(in_channels=self.n_channel,
                                                      out_channels=1,
                                                      kernel_size=self.kernel_size,
                                                      padding=self.padding,
                                                      bias=self.bias),
                                            nn.LeakyReLU(),
                                            )          
        
        self.activation = nn.Sigmoid()

    def forward(self, input_tensor):
        T = input_tensor.shape[1]//self.n_channel
        mask = torch.zeros(input_tensor.shape).to(input_tensor.device)
        for t in range(0,T):
            mask[:,self.n_channel*t:self.n_channel*(t+1),:,:] = self.conv_reduction(input_tensor[:,self.n_channel*t:self.n_channel*(t+1),:,:])
        return self.activation(mask)

class LSTMTimeMask(nn.Module):
    """
    LSTM-based temporal mask.
    Parameters
    ----------
    n_channel: int
        Number of input channels.
    kernel_size: int
        Size of the convolution kernel for reducing image to 1.
    bias: bool
        Whether or not to add the bias.
    """    
    def __init__(self, n_channel=3, kernel_size=3, img_size=(160,120), lstm_layers=3, bias=True,bidirectional=True,):
        super(LSTMTimeMask, self).__init__()
        self.n_channel = n_channel
        self.kernel_size = kernel_size
        self.img_size = img_size
        self.padding = self.kernel_size//2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.n_channel,
                              out_channels=1,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
#        self.conv_reduction = nn.Sequential(
#                                            nn.Conv2d(in_channels=self.n_channel,
#                                                      out_channels=self.n_channel,
#                                                      kernel_size=self.kernel_size,
#                                                      padding=self.padding,
#                                                      bias=self.bias),
#                                            nn.LeakyReLU(),
#                                            nn.Conv2d(in_channels=self.n_channel,
#                                                      out_channels=self.n_channel,
#                                                      kernel_size=self.kernel_size,
#                                                      padding=self.padding,
#                                                      bias=self.bias),
#                                            nn.LeakyReLU(),
#                                            nn.Conv2d(in_channels=self.n_channel,
#                                                      out_channels=1,
#                                                      kernel_size=self.img_size,
#                                                      padding=0,
#                                                      bias=self.bias))

        self.conv_reduction = nn.Sequential(                                            
                                            nn.Conv2d(in_channels=self.n_channel,
                                                      out_channels=1,
                                                      kernel_size=self.img_size,
                                                      padding=0,
                                                      bias=self.bias),
                                            nn.LeakyReLU(),)
        
        self.lstm = nn.LSTM(input_size=3, # Avg, max pool, and convolution?
                            hidden_size=1, # Output to be used for attention
                            num_layers=lstm_layers,
                            bias=bias,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.maxpool = nn.MaxPool1d(kernel_size=self.n_channel)
        self.avgpool = nn.AvgPool1d(kernel_size=self.n_channel)
        self.activation = nn.Sigmoid()
        

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        T = input_tensor.shape[1]//self.n_channel
        conv_value = torch.zeros((batch_size,T)).to(input_tensor.device)
        for t in range(0,T):
            conv_value[:,t] = self.conv_reduction(input_tensor[:,self.n_channel*t:(self.n_channel*(t+1)),:,:])
        avg_value = self.avgpool(torch.mean(torch.mean(input_tensor,dim=-1),dim=-1))
        max_value = self.maxpool(torch.max(torch.max(input_tensor,dim=-1)[0],dim=-1)[0])
        hidden = self.lstm(torch.stack([avg_value,max_value,conv_value],dim=-1))[0]
        attention = self.activation(hidden.sum(dim=-1))
        
        return attention

class CBAMSpaceMask(nn.Module):
    """
    CBAM spatial mask.
    Parameters
    ----------
    n_channel: int
        Number of input channels.
    conv_kernel_size: int
        Size of the convolution kernel.
    pool_kernel_size: int
        Size of the pooling kernels.
    bias: bool
        Whether or not to add the bias.        
    """    
    def __init__(self, n_channel=3, conv_kernel_size=3, pool_kernel_size=3, bias=True):
        super(CBAMSpaceMask, self).__init__()

        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_kernel_size // 2
        self.pool_kernel_size = pool_kernel_size
        self.pool_padding = pool_kernel_size // 2
        self.bias = bias
        
        self.conv_channel = int(n_channel*2) # Avg & max pooled values
        self.n_channel = n_channel        
        
        self.maxpool = nn.MaxPool2d(kernel_size=self.pool_kernel_size,
                                    stride=1,
                                    padding=self.pool_padding)
        self.avgpool = nn.AvgPool2d(kernel_size=self.pool_kernel_size,
                                    stride=1,
                                    padding=self.pool_padding)
        
#        self.conv_reduction = nn.Sequential(
#                                            nn.Conv2d(in_channels=self.conv_channel,
#                                                      out_channels=self.conv_channel,
#                                                      kernel_size=self.conv_kernel_size,
#                                                      padding=self.conv_padding,
#                                                      bias=self.bias),
#                                            nn.LeakyReLU(),
#                                            nn.Conv2d(in_channels=self.conv_channel,
#                                                      out_channels=self.n_channel,
#                                                      kernel_size=self.conv_kernel_size,
#                                                      padding=self.conv_padding,
#                                                      bias=self.bias),
#                                            nn.LeakyReLU(),
#                                            nn.Conv2d(in_channels=self.n_channel,
#                                                      out_channels=1,
#                                                      kernel_size=self.conv_kernel_size,
#                                                      padding=self.conv_padding,
#                                                      bias=self.bias))   
        self.conv_reduction = nn.Sequential(
                                            nn.Conv2d(in_channels=self.conv_channel,
                                                      out_channels=1,
                                                      kernel_size=self.conv_kernel_size,
                                                      padding=self.conv_padding,
                                                      bias=self.bias),
                                            nn.LeakyReLU(),)   
        self.activation = nn.Sigmoid()

    def forward(self, input_tensor):        
        batch_size = input_tensor.shape[0]
        T = input_tensor.shape[1]//self.n_channel
        
        max_value = self.maxpool(input_tensor)
        avg_value = self.avgpool(input_tensor)
        pooled_value = []
        for t in range(0,input_tensor.shape[1]):
            pooled_value.append(max_value[:,t:t+1,:,:])
            pooled_value.append(avg_value[:,t:t+1,:,:])
        
        pooled_value = torch.cat(pooled_value, dim=1)
        
        mask = torch.zeros(input_tensor.shape).to(input_tensor.device)
        
        for t in range(0,T):
            mask[:,self.n_channel*t:self.n_channel*(t+1),:,:] = self.conv_reduction(pooled_value[:,self.conv_channel*t:self.conv_channel*(t+1),:,:])
        
        return self.activation(mask)

    
class CBAMTimeMask(nn.Module):
    """
    CBAM temporal mask.
    Parameters
    ----------
    n_frame: int
        Number of input frames
    n_channel: int
        Number of input channels.
    conv_kernel_size: int
        Size of the convolution kernel.
    pool_kernel_size: int
        Size of the pooling kernels.
    bias: bool
        Whether or not to add the bias.        
    """    
    def __init__(self, n_channel=3, conv_kernel_size=3, pool_kernel_size=3, bias=True):
        super(CBAMTimeMask, self).__init__()

        self.n_channel = n_channel
        self.kernel_size = conv_kernel_size
        self.padding = self.kernel_size//2
        self.bias = bias
        
        self.conv = nn.Sequential(nn.Conv1d(in_channels=2, # Avg & max pool
                                      out_channels=1,
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      bias=self.bias),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(in_channels=1, # Avg & max pool
                                      out_channels=1,
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      bias=self.bias),
                                     nn.LeakyReLU())

        self.maxpool = nn.MaxPool1d(kernel_size=self.n_channel)
        self.avgpool = nn.AvgPool1d(kernel_size=self.n_channel)
        self.activation = nn.Sigmoid()
                
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        avg_value = self.avgpool(torch.mean(torch.mean(input_tensor,dim=-1),dim=-1))
        max_value = self.maxpool(torch.max(torch.max(input_tensor,dim=-1)[0],dim=-1)[0])
                
        pooled_values = torch.stack([avg_value,max_value],dim=1)
        attention = self.activation(self.conv(pooled_values)).squeeze(1)
        return attention
        

def reshape_attention(attention, n_channel, H, W):
    batch_size = attention.shape[0]
    new_attention = attention.repeat_interleave(n_channel,dim=1).reshape(batch_size,attention.shape[1]*n_channel,1,1)
    new_attention = new_attention.repeat_interleave(H,dim=2).repeat_interleave(W,dim=3)
    return new_attention


# Attention should be 1-other since we are using sigmoid rather than softmax. 
class CBAMVideoNetwork(nn.Module):
    """
    CBAM-based video network.
    Parameters
    ----------
    SpatialBlockAbnormal:
        Spatial mask block for abnormal side
    TemporalBlockAbnormal: 
        Temporal mask block for abnormal side
    SpatialBlockNormal:
        Spatial mask block for normal side
    TemporalBlockNormal: 
        Temporal mask block for normal side        
    """    
    def __init__(self, SpatialBlockType=1, TemporalBlockType=1, kernel_size=3,n_channel=3,bias=True):
        super(CBAMVideoNetwork, self).__init__()
        
        if SpatialBlockType == 1:
            self.spatial_abnormal = ConvSpaceMask()
            self.spatial_normal = ConvSpaceMask()
        else:
            self.spatial_abnormal = CBAMSpaceMask()
            self.spatial_normal = CBAMSpaceMask()
        
        if TemporalBlockType == 1:
            self.temporal_abnormal = LSTMTimeMask()
            self.temporal_normal = LSTMTimeMask()
        else:
            self.temporal_abnormal = CBAMTimeMask()
            self.temporal_normal = CBAMTimeMask()            
            
        self.n_channel = n_channel
        self.kernel_size = kernel_size
        self.bias = bias
        self.fc_conv_abnormal = nn.Sequential(
                                    nn.Conv2d(in_channels=self.n_channel,
                                              out_channels=1,
                                              kernel_size=self.kernel_size,
                                              padding=0,
                                              bias=self.bias),
                                    nn.MaxPool2d(kernel_size=2),)  
        self.fc_conv_normal = nn.Sequential(
                                    nn.Conv2d(in_channels=self.n_channel,
                                              out_channels=1,
                                              kernel_size=self.kernel_size,
                                              padding=0,
                                              bias=self.bias),
                                    nn.MaxPool2d(kernel_size=2),)           
        
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        T, H, W = input_tensor.shape[1]//self.n_channel, input_tensor.shape[2], input_tensor.shape[3]
        spatial_attention_abnormal = self.spatial_abnormal(input_tensor)
        temporal_attention_abnormal = self.temporal_abnormal(input_tensor)
        spatial_attention_normal = self.spatial_normal(input_tensor)
        temporal_attention_normal = self.temporal_normal(input_tensor)    
        
        temporal_attention_abnormal_reshaped = reshape_attention(temporal_attention_abnormal,self.n_channel,H,W)
        temporal_attention_normal_reshaped = reshape_attention(temporal_attention_normal,self.n_channel,H,W)
        
        masked_tensor_abnormal = input_tensor*spatial_attention_abnormal*temporal_attention_abnormal_reshaped
        masked_tensor_normal = input_tensor*spatial_attention_normal*temporal_attention_normal_reshaped

        abnormal_sum = torch.zeros((batch_size,self.n_channel,masked_tensor_abnormal.shape[2],masked_tensor_abnormal.shape[3])).to(input_tensor.device)
        normal_sum = torch.zeros((batch_size,self.n_channel,masked_tensor_normal.shape[2],masked_tensor_normal.shape[3])).to(input_tensor.device)
        
        for t in range(0,T):
            abnormal_sum += masked_tensor_abnormal[:,self.n_channel*t:self.n_channel*(t+1),:,:]
            normal_sum += masked_tensor_normal[:,self.n_channel*t:self.n_channel*(t+1),:,:]
        
        abnormal_score = torch.sum(self.fc_conv_abnormal(abnormal_sum),dim=(1,2,3))
        normal_score = torch.sum(self.fc_conv_normal(normal_sum),dim=(1,2,3))
        
        return abnormal_score, normal_score, temporal_attention_abnormal, temporal_attention_normal
    
class AttentionMatchLoss(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super(AttentionMatchLoss, self).__init__()
        self.eps = eps
        self.celoss = nn.CrossEntropyLoss()
        
    def forward(self, label, abnormal_score,normal_score,temporal_attention_abnormal,temporal_attention_normal):
        loss = self.celoss(label, torch.stack([abnormal_score,normal_score],dim=-1))
        inverted_attention_abnormal = 1-temporal_attention_abnormal
        loss += torch.mean((inverted_attention_abnormal-temporal_attention_normal)**2)

        return loss

    
class ViolenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, frame_size=(160,120), n_channel=3,is_test=False,normalize=False, sampling_freq=1):
        self.fight_dir = os.path.join(data_path,'frames/fight')
        self.normal_dir = os.path.join(data_path,'frames/normal')
        self.test_file = os.path.join(data_path,'test_videos.csv')
        
        test_videos = list(np.loadtxt(self.test_file,delimiter=',',dtype='str'))
        fight_videos = [video for video in os.listdir(self.fight_dir) if video not in test_videos]
        normal_videos = [video for video in os.listdir(self.normal_dir) if video not in test_videos]
        
        self.test_labels = [1 if video[0] == 'F' else 0 for video in test_videos]
        self.fight_labels = [1 if video[0] == 'F' else 0 for video in fight_videos]
        self.normal_labels = [1 if video[0] == 'F' else 0 for video in normal_videos]
        
        self.frame_size = frame_size
        self.n_channel = n_channel
        self.is_test = is_test
        self.sampling_freq = sampling_freq
        
        if self.is_test:
            self.actual_videos = [os.path.join(self.fight_dir,video) if video[0]=='F' else os.path.join(self.normal_dir,video) for video in test_videos]
            self.actual_labels = self.test_labels
        else:
            self.fight_videos = [os.path.join(self.fight_dir,video) if video[0]=='F' else os.path.join(self.normal_dir,video) for video in fight_videos]
            self.normal_videos = [os.path.join(self.fight_dir,video) if video[0]=='F' else os.path.join(self.normal_dir,video) for video in normal_videos]
            self.actual_videos = self.fight_videos+self.normal_videos 
            self.actual_labels = self.fight_labels+self.normal_labels
            
        if normalize:
            self.normalizer = 1/255.0
        else:
            self.normalizer = 1
        
    def __len__(self) -> int:
        return len(self.actual_videos)
            
    def __getitem__(self, idx):
        video_name = self.actual_videos[idx]
        frame_count = len(os.listdir(video_name))
        video = []
        for count in range(0,frame_count,self.sampling_freq):
            video.append(cv2.imread(os.path.join(video_name,'{0}.jpg'.format(count))))
        video = self.normalizer*np.concatenate(video,axis=2)   
        label = torch.zeros(2)
        label[self.actual_labels[idx]] = 1
        return torch.Tensor(video).permute(2,0,1), label
        
        
        
        
        
        

