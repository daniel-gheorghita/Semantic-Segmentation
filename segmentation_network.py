import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

class SegmentationNetwork(nn.Module):

    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 240, 240), num_filters=5, kernel_size=3,
                 stride=1, weight_scale=0.001, pool=2, hidden_dim=20,
                 num_classes=23, dropout=0.0, use_resnet18 = False, ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: The size of the window to take a max over.
        - weight_scale: Scale for the convolution weights initialization-
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(SegmentationNetwork, self).__init__()
        self.channels, self.height, self.width = input_dim
        self.use_resnet18 = use_resnet18
        
        ############################################################################
        # TODO: Initialize the necessary layers to resemble the ThreeLayerCNN      #
        # architecture  from the class docstring. In- and output features should   #
        # not be hard coded which demands some calculations especially for the     #
        # input of the first fully convolutional layer. The convolution should use #
        # "same" padding which can be derived from the kernel size and its weights #
        # should be scaled. Layers should have a bias if possible.                 #
        ############################################################################
        
        
        
        if (self.use_resnet18 == True):
            self.intern_model = models.resnet18(pretrained=True)
            #Do not retrain the model
            for param in self.intern_model.parameters():    
                param.requires_grad = False
            print 'Resnet18 initialized.'
            
            self.initializationStarted = 0
            
            num_ftrs = self.intern_model.fc.in_features
            self.intern_model.fc = None #nn.Linear(1, 1) 
            self.intern_model.layer4 = None #nn.Linear(1, 1) 
            #self.intern_model.layer3 = None #nn.Linear(1, 1) 
            #for param in self.intern_model.fc.parameters():
                #param.requires_grad = False
            
            
            # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            
            #self.deconvLayer = nn.ConvTranspose2d(,3,60)
            #self.deconvLayer2 = nn.ConvTranspose2d(3,5,40)
            #self.deconvLayer3 = nn.ConvTranspose2d(5,23,23)
            
        else:
            
            pad = 1
            self.pool = pool
            self.num_filters = num_filters
            H_out = 1 + (self.height + 2 * pad - kernel_size) / stride
            W_out = 1 + (self.width + 2 * pad - kernel_size) / stride
            H_out2 = 1 + (H_out + 2 * pad - kernel_size) / stride
            W_out2 = 1 + (W_out + 2 * pad - kernel_size) / stride
            """
            self.convLayer = nn.Conv2d(self.channels,num_filters,kernel_size,stride,pad)
            #ReLU 
            self.maxPoolLayer = nn.MaxPool2d(pool,pool)
            self.fc1Layer = nn.Linear(num_filters*((W_out-pool)/pool+1) * ((H_out-pool)/pool+1), hidden_dim)
            self.dropoutLayer = nn.Dropout2d(dropout)
            #ReLU
            self.fc2Layer = nn.Linear( hidden_dim  , self.height * self.width * num_classes)
            """
            
            self.convLayer = nn.Conv2d(self.channels,num_filters,kernel_size,stride,pad)
            #ReLU 
            self.maxPoolLayer = nn.MaxPool2d(pool,pool)
            self.conv2Layer = nn.Conv2d(num_filters,num_filters*2,kernel_size,stride,pad)
            #self.fc1Layer = nn.Linear(num_filters*((W_out-pool)/pool+1) * ((H_out-pool)/pool+1), hidden_dim)
            self.dropoutLayer = nn.Dropout2d(dropout)
            #ReLU
            
            
            self.deconvLayer = nn.ConvTranspose2d(num_filters*2,3,60)
            self.deconv2Layer = nn.ConvTranspose2d(3,5,40)
            self.deconv3Layer = nn.ConvTranspose2d(5,23,23)
            #self.fc2Layer = nn.Linear( hidden_dim  , self.height * self.width * num_classes)
            #self.conv2Layer = nn.Conv2d(self.channels,num_filters,kernel_size,stride,pad)
            ############################################################################
            #                             END OF YOUR CODE                             #
            ############################################################################
    def forward_for_init(self, x):    
        (N, C,Hinput,Winput) = x.data.size()
        x = self.intern_model.conv1(x)
        x = self.intern_model.bn1(x)
        x = self.intern_model.relu(x)
        x = self.intern_model.maxpool(x)

        x = self.intern_model.layer1(x)
        
        
        
        
        x = self.intern_model.layer2(x)
        if (self.initializationStarted == 0):
        #print x.data.size()
            print 'Deconv2 initialized.'
            (N2,C2,H2,W2) = x.data.size()
            #print x.data.size()
            #self.deconvLayer2 = nn.ConvTranspose2d(C2,23,kernel_size=(Hinput-H2+1,Winput-W2+1))
        
        
        
        
        
        x = self.intern_model.layer3(x)
            
            
        #x = self.intern_model.layer4(x)
        #print 'Deconv 3 not initialized.'
        if (self.initializationStarted == 0):
        #print x.data.size()
            print 'Deconv3 initialized.'
            (N3,C3,H3,W3) = x.data.size()
            #print x.data.size()
            self.deconvLayer3 = nn.ConvTranspose2d(C3,23,kernel_size=(Hinput-H3+1,Winput-W3+1))
            self.initializationStarted = 1
                
        x = self.deconvLayer3(x)
        
        
        
        
        
    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ############################################################################
        # TODO: Chain our previously initialized convolutional neural network      #
        # layers to resemble the architecture drafted in the class docstring.      #
        # Have a look at the Variable.view function to make the transition from    #
        # convolutional to fully connected layers.                                 #
        ############################################################################
        #x.data = x.data[:,:,0:120,0:120]
        (N, C,Hinput,Winput) = x.data.size()
        #print x.data.size()
        
        if (self.use_resnet18 == True):
            #x = self.intern_model(x)
            #print x.data.size()
            
             # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            
            #self.deconvLayer = nn.ConvTranspose2d(,3,60)
            #self.deconvLayer2 = nn.ConvTranspose2d(3,5,40)
            #self.deconvLayer3 = nn.ConvTranspose2d(5,23,23)
            
            
            x = self.intern_model.conv1(x)
            #x = self.intern_model.bn1(x)
            x = self.intern_model.relu(x)
            x = self.intern_model.maxpool(x)

            x = self.intern_model.layer1(x)
            x = self.intern_model.layer2(x)
            
            #y = self.deconvLayer2(x)

            
            
            x = self.intern_model.layer3(x)
            
            
            #x = self.intern_model.layer4(x)
                
            x = self.deconvLayer3(x)
            #print x.data.size()
            
            #x = self.avgpool(x)
            #x = x.view(x.size(0), -1)
            #x = self.fc(x)


        else:
            """
            x = self.convLayer(x)
            #x = F.relu(x)
            x = self.maxPoolLayer(x)
            
            (_, C, H, W) = x.data.size()
            x = x.view( -1 , C * H * W)

            x = self.fc1Layer(x)
            x = self.dropoutLayer(x)
            #x = F.relu(x)
            x = self.fc2Layer(x)
            """
            x = self.convLayer(x)
            #x = F.relu(x)
            x = self.maxPoolLayer(x)
            x = self.conv2Layer(x)
            #x = self.conv2Layer(x)
            x = self.dropoutLayer(x)
            #x = F.relu(x)
            #print x.data.size()
            x = self.deconvLayer(x, output_size = (N,5,179,179))
            #print x.data.size()
            x = self.deconv2Layer(x,output_size = (N,5,218,218))
            #print x.data.size()
            x = self.deconv3Layer(x,output_size = (N,23,240,240))
            #print x.data.size()
            
        #print x.size()
        #x = x.view(N,23,Hinput,Winput)
        return x#+y
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)