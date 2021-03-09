#--------------------------------------------#
#   This part of the code is only used to see the network structure, not the test code
#--------------------------------------------#
from nets.pspnet import pspnet

if __name__ == "__main__":
    model = pspnet(21,[473,473,3],downsample_factor=16,backbone='mobilenet',aux_branch=False)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
