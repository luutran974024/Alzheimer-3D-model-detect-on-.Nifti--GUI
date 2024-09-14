# Image_Recognition_WebGUI

✨ **Alzheimer intelligent diagnosis web application based on 3D convolutional neural network and ADNI data set**: AI artificial intelligence image recognition-Pytorch; Visual Web graphical interface-Pywebio; nii medical image recognition. 100% pure Python

🚩[English Readme](./README.en.md)


🔔 If you have any project-related questions, you are welcome to raise an `issue` in this project, and I will usually respond within 24 hours.

## Function introduction

- 1. Intelligent diagnosis of Alzheimer’s disease based on brain MRI medical images
- 2. Draw parameter correlation heat map
- 3. Written in pure python, lightweight, easy to reproduce, and easy to deploy
- 4. The code is highly readable and the core parts have extremely detailed comments.

## Interface display
- Enter the web interface
 ![image](./readme_static/readme_img/4.png)
- Click "Use demo.nii" to use the default demo image to test the recognition function
 ![image](./readme_static/readme_img/3.png)
- You can also upload medical images yourself
 ![image](./readme_static/readme_img/9.png)
- Click "View Image" to render parameter heat map
 ![image](./readme_static/readme_img/5.png)
 ![image](./readme_static/readme_img/6.png)
- Generate parameter correlation heat map based on uploaded images
 ![image](./readme_static/readme_img/7.png)

## how to use

python version 3.9

Requires `8GB` or more memory

Install dependencies first

```bash
pip install -r requirement.txt
```

demo01.py It is the project entrance. Run this file to start the server.
```bash
python demo01.py
```

Copy the link and open it in your browser
![image](./readme_static/readme_img/10.png)
Click "Demo" to enter the web interface
![image](./readme_static/readme_img/11.png)

Afterwards, you can click "Use demo.nii" to use the default test example. You can also click "Upload.nii" and select image files of different categories in the readme_static/test folder to upload for testing.

## Project structure

```
└─Image_Recognition_WebGUI
    ├─data
    │  └─model_save
    ├─imgs
    │  ├─img_hot
    │  ├─img_merge
    │  └─img_raw
    ├─nii
    ├─readme_static
    │  ├─readme_img
    │  └─test
    │      ├─AD
    │      ├─CN
    │      ├─EMCI
    │      ├─LMCI
    │      └─MCI
    └─run_logs
```

- The data folder stores some static resources, and the model_save folder stores the trained model.
- The imgs folder stores rendered images
- The nii folder stores medical imaging data uploaded by users
- readme_static stores static resources used in readme documents
- The readme_static/test folder stores some image files of five categories, which can be used for testing
- run_logs stores user access logs
## Classifier core code

```python
from torch import nn
import torch

class ClassificationModel3D(nn.Module):
      """Classifier model"""    
      def __init__(self, dropout=0.4, dropout2=0.4):
        nn.Module.__init__(self)

        # 定义四个Conv3d层
        self.Conv_1 = nn.Conv3d(1, 8, 3)  #The number of input channels is 1, the number of output channels is 8, and the convolution kernel size is 3x3x3
        self.Conv_2 = nn.Conv3d(8, 16, 3)  #The number of input channels is 8, the number of output channels is 16, and the convolution kernel size is 3x3x3
        self.Conv_3 = nn.Conv3d(16, 32, 3)  # The number of input channels is 16, the number of output channels is 32, and the convolution kernel size is 3x3x3
        self.Conv_4 = nn.Conv3d(32, 64, 3)  # The number of input channels is 32, the number of output channels is 64, and the convolution kernel size is 3x3x3
        # Define four BatchNorm3d layers, each convolutional layer is followed by a BatchNorm3d layer        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_4_bn = nn.BatchNorm3d(64)

        # Define four MaxPool3d layers, each convolutional layer is followed by a MaxPool3d layer
        self.Conv_1_mp = nn.MaxPool3d(2)  # 池化核大小为2
        self.Conv_2_mp = nn.MaxPool3d(3)  # 池化核大小为3
        self.Conv_3_mp = nn.MaxPool3d(2)  # 池化核大小为2
        self.Conv_4_mp = nn.MaxPool3d(3)  # 池化核大小为3

        # Define two fully connected layers
        self.dense_1 = nn.Linear(4800, 128)  # 输入维度为4800，输出维度为128
        self.dense_2 = nn.Linear(128, 5)  # 输入维度为128，输出维度为5。因为这是一个五分类问题，所以最终需要输出维度为5

        # Define ReLU activation function and dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        self.dropout2 = nn.Dropout(dropout2)  # 增强鲁棒性

    def forward(self, x):
       def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
      """This line of code performs convolution, batch normalization, and a ReLU activation function on the input x.

      self.Conv_1(x) performs a 3D convolution operation on the input x and outputs a feature map.

      self.Conv_1_bn(...) performs a batch normalization operation on the feature map output by the convolution to obtain the normalized feature map.

      self.relu(...) performs the ReLU activation function operation on the normalized feature map to obtain the activated feature map.

      What the entire operation does is extract features from the input x and nonlinearize them so that the network can better learn these features. Batch normalization technology is used here, which can speed up the training process of the model and improve the generalization ability of the model. The final output result is the feature map x processed by convolution, batch normalization and ReLU activation function.
 """
       # Maximum pooling of the first convolutional layer
        x = self.Conv_1_mp(x)
        """
        This line of code performs max pooling on the input x, reducing the size of the feature map by half.

        self.Conv_1_mp(...) performs a max pooling operation on the input x, with a pooling kernel size of 2.

        The pooling operation extracts the maximum value within each pooling window in the feature map as the value of the corresponding position of the output feature map, thus reducing the size of the feature map by half.

        The max pooling operation can help the network achieve spatial invariance, allowing the network to still recognize the same features when the input changes slightly. In this model, the feature map x after max pooling is passed to the next convolutional layer for feature extraction and nonlinearization.
 """
        # The second convolutional layer
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        # Maximum pooling of the second convolutional layer
        x = self.Conv_2_mp(x)
        # The third convolutional layer
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        # Maximum pooling of the third convolutional layer
        x = self.Conv_3_mp(x)
        # The fourth convolutional layer
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        # Maximum pooling of the fourth convolutional layer
        x = self.Conv_4_mp(x)
        # Flatten the tensor into a one-dimensional vector
        x = x.view(x.size(0), -1)
        """
        This line of code flattens the input tensor x into a one-dimensional vector.

        x.size(0) gets the size of the first dimension of the input tensor x, which is the batch size of the tensor.

        -1 means flatten the second dimension and all dimensions after it into one dimension.

        x.view(...) performs a shape transformation on the input tensor x, flattening it into a one-dimensional vector.

        The function of this operation is to turn the feature map x after convolution and pooling into a one-dimensional vector, so that it can be passed to the fully connected layer for tasks such as classification or regression. The flattened vector size is (batch_size, num_features), where batch_size is the batch size of the input tensor, and num_features is the number of flattened vector elements, which is the number of features after convolution and pooling.
        """
        # dropout layer
        x = self.dropout(x)
        """
        This line of code performs a dropout operation on the input tensor x, that is, setting some elements in the input tensor to zero with a certain probability.

        self.dropout(...) performs a dropout operation on the input tensor x, with a dropout probability of dropout.

        The dropout operation will set some elements in the input tensor to zero with a certain probability, thereby achieving the purpose of random deactivation. This can reduce overfitting and enhance the generalization ability of the model.

        In this model, the dropout operation is applied before the fully connected layer, which can help the model better learn the characteristics of the data and prevent overfitting. The final x tensor is the result of the dropout operation and will be passed to the next fully connected layer for processing.
        """
        # Fully connected layer 1
        x = self.relu(self.dense_1(x))
        """
        This line of code performs a fully connected operation on the input tensor x and applies the ReLU activation function.

        self.dense_1(x) performs a fully connected operation on the input tensor x, mapping it into a feature space of size 128.

        self.relu(...) performs the ReLU activation function on the output of the fully connected layer to obtain the activated feature vector.

        In this model, the role of the fully connected layer is to map the feature vectors processed by convolution, pooling and dropout into a new feature space to facilitate tasks such as classification or regression. The function of the ReLU activation function is to perform nonlinear processing on the feature vector, so that the network can better learn the nonlinear correlation in the data. The final x tensor is the result of processing by the fully connected layer and the ReLU activation function, and will be passed to the next dropout layer for processing.
        """
        # dropout2 layer
        x = self.dropout2(x)
        # Fully connected layer 2
        x = self.dense_2(x)
        # Return the output result
        return x


        if __name__ == "__main__":
        #Create an instance model of the ClassificationModel3D class, that is, create a 3D image classification model
        model = ClassificationModel3D()

        #Create a test tensor test_tensor with shape (1, 1, 166, 256, 256),
        # Where 1 represents the batch size, 1 represents the number of input channels, 166, 256 and 256 represent the depth, height and width of the input data respectively
        test_tensor = torch.ones(1, 1, 166, 256, 256)

        # Perform forward pass on the test tensor test_tensor to get the output of the model output
        output = model(test_tensor)

        #Print the shape of the output result, that is, (batch_size, num_classes), where batch_size is the batch size of the test tensor and num_classes is the number of categories of the classification task
        print(output.shape)

```

ref: https://github.com/moboehle/Pytorch-LRP

Dataset: https://adni.loni.usc.edu

# Open source license

This translated version is for reference only, the English version in the LICENSE file shall prevail

MIT Open Source License:

Copyright (c) 2023 bytesc

The right to use, copy, modify, merge, publish, distribute, sublicense and/or sell the Software is hereby granted, free of charge, to any person obtaining a copy of this software and related documentation files (the "Software"), subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions thereof.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. In no event shall the author or copyright holder be liable for any claim, damages or other liability, whether in contract, tort or otherwise, arising out of the use of this software.