from keras import layers
from keras import models
from keras.layers import Conv3D,MaxPooling3D
from keras.optimizers import SGD,RMSprop
import numpy as np
#
# image dimensions
#

img_height = 112
img_width = 112
img_depth = 16
img_channels = 3

#
# network params
#
cardinality=32
'''train_X = np.load('train_X.npy')
train_Y = np.load('train_Y.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')

num_classes = 14

X = []

for i in range(train_X.shape[0]):
    z = train_X[i].shape[0]
    
    if z < 498:
        temp = np.zeros(shape = (498,64,64))
        temp[0:z,:,:] = train_X[i][0]
        train_X[i] = temp
    X.append(train_X[i])

X = np.array(X)
train_X = X


X = []
for i in range(test_X.shape[0]):
    z = test_X[i].shape[0]
    if z < 498:
        temp = np.zeros(shape = (498,64,64))
        temp[0:z,:,:] = test_X[i]
        test_X[i] = temp
    X.append(test_X[i])

test_X = np.array(X)
train_Y = np.reshape(train_Y,(-1,14))
test_Y = np.reshape(test_Y,(-1,14))
train_X = train_X[:,50:300,16:48,16:48]
test_X = test_X[:,50:300,16:48,16:48]

print(test_X.shape,train_X.shape)

train_X = np.reshape(train_X,(-1,250,32,32,1))
test_X = np.reshape(test_X,(-1,250,32,32,1))
shape=(250,32,32,1)
train_X = np.concatenate((train_X,test_X),axis = 0)
train_Y = np.concatenate((train_Y,test_Y),axis = 0)
print(test_X.shape,train_X.shape)
'''
def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        print(y.shape)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv3D(nb_channels, kernel_size=(3, 3,3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:,:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv3D(_d, kernel_size=(3, 3,3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)


        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1,1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv3D(nb_channels_in, kernel_size=(1, 1,1), strides=(1, 1,1), padding='same')(y)

        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv3D(nb_channels_out, kernel_size=(1, 1,1), strides=(1, 1,1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1,1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv3D(nb_channels_out, kernel_size=(1, 1,1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv3D(64, kernel_size=(7, 7,7), strides=(2, 2,2), padding='same')(x)
    print(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool3D(pool_size=(3, 3,3), strides=(2, 2,2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2,2) if i == 0 else (1, 1,1)
        x = residual_block(x, 256, 512, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2,2) if i == 0 else (1, 1,1)
        x = residual_block(x, 512, 1024, _strides=strides)
    print(x)
    # conv5
    for i in range(3):
        strides = (2, 2,2) if i == 0 else (1, 1,1)
        x = residual_block(x, 1024, 2048, _strides=strides)
    print(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(14)(x)

    return x


image_tensor = layers.Input(shape=(img_height, img_width,img_depth, img_channels))
network_output = residual_network(image_tensor)
  
model = models.Model(inputs=[image_tensor], outputs=[network_output])
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])

model.fit(train_X,train_Y,batch_size=4,epochs=10,shuffle=True,validation_data=(test_X,test_Y),verbose=1)
model.save('my_model')

print(model.summary())
