# Copyright (c) 2024 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clauseimport argparse

import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model implemented using PyTorch.

    Args:
        size_in (int): Number of input features.
        size_out (int): Number of output features.
        hidden_size (int): Number of neurons in the hidden layer.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        activation (nn.LeakyReLU): Leaky ReLU activation function with a negative slope of 0.1.
        batchnorm (nn.BatchNorm1d): Batch normalization layer for the hidden layer.
        dropout (nn.Dropout): Dropout layer with a dropout probability of 0.5.

    Methods:
        forward(x): Forward pass of the model.

    Note:
        This MLP consists of two fully connected layers with a Leaky ReLU activation function,
        batch normalization, and dropout applied to the hidden layer. It is designed for general-purpose
        regression or classification tasks.

    Example:
        >>> model = MLP(size_in=100, size_out=10, hidden_size=50)
        >>> input_data = torch.randn(32, 100)  # Batch size of 32, input features of size 100
        >>> output = model(input_data)
    """

    def __init__(self, size_in, size_out, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(size_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, size_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.batchnorm(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return y
    
class ConvBlock(nn.Module):
    """
    A Convolutional Block module implemented using PyTorch.

    Args:
        in_channels (int): Number of input channels for the convolutional layer.
        out_channels (int): Number of output channels for the convolutional layer.
        kernel_size (int): Size of the convolutional kernel.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        batchnorm (nn.BatchNorm2d): Batch normalization layer for the output channels.
        leakyrelu (nn.LeakyReLU): Leaky ReLU activation function with a negative slope of 0.1.
        dropout (nn.Dropout2d): 2D dropout layer with a dropout probability of 0.5.
        maxpool (nn.MaxPool2d): 2D Max pooling layer with a kernel size of 2 and stride of 2.

    Methods:
        forward(x): Forward pass of the ConvBlock.

    Note:
        This ConvBlock consists of a convolutional layer followed by batch normalization, Leaky ReLU activation,
        dropout, and max pooling. It is designed for use in convolutional neural network architectures.

    Example:
        >>> conv_block = ConvBlock(in_channels=3, out_channels=64, kernel_size=3)
        >>> input_data = torch.randn(32, 3, 64, 64)  # Batch size of 32, 3 input channels, image size 64x64
        >>> output = conv_block(input_data)
    """
    def __init__(self, in_channels=32, out_channels=32, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same")
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.conv(x)
        y = self.batchnorm(y)
        y = self.leakyrelu(y)
        y = self.dropout(y)
        y = self.maxpool(y)
        return y
    
class ConvEncoder(nn.Module):
    """
    Convolutional Encoder module implemented using PyTorch.

    Args:
        size_in (int): Number of input channels for the first convolutional block.
        size_out (int): Number of output channels for each convolutional block.

    Attributes:
        conv_block1 (ConvBlock): First convolutional block.
        conv_block2 (ConvBlock): Second convolutional block.
                                [...]
        conv_block7 (ConvBlock): Seventh convolutional block.

    Methods:
        forward(x): Forward pass of the ConvEncoder.

    Note:
        This ConvEncoder consists of multiple ConvBlock layers to progressively reduce spatial dimensions.
        It is designed for encoding and feature extraction in convolutional neural network architectures.

    Example:
        >>> conv_encoder = ConvEncoder(size_in=3, size_out=32)
        >>> input_data = torch.randn(32, 3, 64, 64)  # Batch size of 32, 3 input channels, image size 64x64
        >>> encoded_output = conv_encoder(input_data)
    """
    def __init__(self, size_in=3, size_out=32):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels=size_in, out_channels=size_out)
        self.conv_block2 = ConvBlock(in_channels=size_out, out_channels=size_out)
        self.conv_block3 = ConvBlock(in_channels=size_out, out_channels=size_out)
        self.conv_block4 = ConvBlock(in_channels=size_out, out_channels=size_out)
        self.conv_block5 = ConvBlock(in_channels=size_out, out_channels=size_out)
        self.conv_block6 = ConvBlock(in_channels=size_out, out_channels=size_out)
        self.conv_block7 = ConvBlock(in_channels=size_out, out_channels=size_out)

    def forward(self, x):
        y = self.conv_block1(x)
        y = self.conv_block2(y)
        y = self.conv_block3(y)
        y = self.conv_block4(y)
        y = self.conv_block5(y)
        y = self.conv_block6(y)
        y = self.conv_block7(y)
        return y
    
class MultiMagnificationNet(nn.Module):
    """
    Multi-Magnification Neural Network inspired by https://doi.org/10.1016/j.jnucmat.2020.152082.

    Args:
        num_levels (int): Number of magnification levels.
        num_channels (int): Number of input channels.
        size_hidden (int): Number of hidden channels in the encoding modules.
        share_weights (bool): Whether to share weights among encoding modules.
        use_mlp (bool): Whether to use an MLP as a classifier or a 1x1 convolutional layer.
        align_features (bool): Whether to physically align features when using joint encodings.

    Attributes:
        num_levels (int): Number of magnification levels.
        num_channels (int): Number of input channels.
        size_hidden (int): Number of hidden channels in the encoding modules.
        share_weights (bool): Whether to physically align features when using joint encodings.
        use_mlp (bool): Whether to use an MLP as a classifier or a 1x1 convolutional layer.
        align_features (bool): Whether to align features using joint encodings.
        encoder (ConvEncoder or nn.ModuleList): Convolutional encoder modules.
        padder (nn.ZeroPad2d): Zero-padding layer for alignment of features.
        classifier (MLP or nn.Conv2d): Classifier for the multi-magnification network.

    Methods:
        forward(x): Forward pass of the MultiMagnificationNet.

    Note:
        This MultiMagnificationNet consists of convolutional encoding modules, a classifier, and optional
        pysical alignment of features using joint encodings. It is designed for multi-magnification image classification.

    Example:
        >>> multi_magnification_net = MultiMagnificationNet(num_levels=4, num_channels=4, size_hidden=128, share_weights=False, use_mlp=True, align_features=True)
        >>> input_data = torch.randn(32, 16, 128, 128)  # Batch size of 32, 4 input channels per magnification level, image size 128x128
        >>> output = multi_magnification_net(input_data) # Output of size 32x1x1x1 with a dense classifier or 32x1 with an MLP classifier
    """
    def __init__(self, num_levels=4, num_channels=4, size_hidden=128, share_weights=False, use_mlp=False, align_features=False):   
        super(MultiMagnificationNet, self).__init__()

        # Config attributes
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.size_hidden = size_hidden
        self.share_weights = share_weights
        self.use_mlp = use_mlp
        self.align_features = align_features

        # Convolutional encoding modules
        if self.align_features:
            if self.share_weights:
                self.encoder = ConvEncoder(size_in=self.num_channels+self.size_hidden, size_out=self.size_hidden)
            else:
                self.encoder = nn.ModuleList(
                    [ConvBlock(in_channels=self.num_channels+self.size_hidden, out_channels=self.size_hidden) for i in range(self.num_levels-1)]
                    )
                self.encoder.append(ConvEncoder(size_in=self.num_channels+self.size_hidden, size_out=self.size_hidden))
            self.padder = nn.ZeroPad2d(int(self.size_hidden/4))
        else:
            if self.share_weights:
                self.encoder = ConvEncoder(size_in=self.num_channels, size_out=self.size_hidden)
            else:
                self.encoder = nn.ModuleList([ConvEncoder(size_in=self.num_channels, size_out=self.size_hidden) for i in range(self.num_levels)])

        # Classifier
        if self.align_features:
            if self.use_mlp:
                self.classifier = MLP(size_in=self.size_hidden, size_out=1, hidden_size=self.size_hidden)
            else:
                self.classifier = nn.Conv2d(in_channels=self.size_hidden, out_channels=1, kernel_size=1)
        else:
            if self.use_mlp:
                self.classifier = MLP(size_in=self.num_levels*self.size_hidden, size_out=1, hidden_size=self.size_hidden)
            else:
                self.classifier = nn.Conv2d(in_channels=self.num_levels*self.size_hidden, out_channels=1, kernel_size=1)
   
    def forward(self, x):
        # Empty tensor for encoding 
        if self.align_features:
            joint_encodings = torch.zeros(x.size(0),self.num_channels+self.size_hidden,x.size(2),x.size(3), device=x.device)
        else:
            joint_encodings = torch.ones(x.size(0),0,1,1, device=x.device)

        # Process patch content
        if self.align_features:
            for i in range(self.num_levels):
                joint_encodings[:, :self.num_channels, :, :] = x[:,self.num_channels*i:self.num_channels*(i+1), :, :]
                if i<self.num_levels-1:
                    if self.share_weights:
                        encodings = self.encoder.conv_block1(joint_encodings)

                    else:
                        encodings = self.encoder[i](joint_encodings)

                    joint_encodings[:,self.num_channels::, :, :] = self.padder(encodings)
                else:
                    if self.share_weights:
                        joint_encodings = self.encoder(joint_encodings)
                    else: 
                        joint_encodings = self.encoder[i](joint_encodings)
        else:
            for i in range(self.num_levels):
                if self.share_weights:
                    encodings = self.encoder(x[:,self.num_channels*i:self.num_channels*(i+1), :, :])
                else:
                    encodings = self.encoder[i](x[:,self.num_channels*i:self.num_channels*(i+1), :, :])
                joint_encodings = torch.cat([joint_encodings, encodings], dim=1)

        # Classify
        if self.use_mlp:
            y = self.classifier(joint_encodings.squeeze())
        else:
            y = self.classifier(joint_encodings)
        return y
    
def main():
    """
    Iterates over model architectures to log number of parameters in the console
    """
    for align_features in [False, True]:
        for use_mlp in [False, True]:
            for share_weights in [False, True]:
                print(
                    f"Model architecture : {'' if align_features else 'no '}features alignment, "
                    f"a {'multi layer' if use_mlp else 'dense layer'} classifier, and "
                    f"{'' if share_weights else 'no '}weight sharing\n"
                    )
                net = MultiMagnificationNet(
                    num_levels=4, 
                    num_channels=4, 
                    size_hidden=128, 
                    share_weights=share_weights, 
                    use_mlp=use_mlp, 
                    align_features=align_features
                    )
                print(net)
                total_params = 0
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        num_params = param.numel()
                        total_params += num_params
                print(f"Total number of parameters :\t{total_params:,}\n\n")
                
if __name__ == "__main__":
    main()