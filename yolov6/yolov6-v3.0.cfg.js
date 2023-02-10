[net]
batch=1
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000
max_batches = 2000200
policy=steps
steps=1600000,1800000
scales=.1,.1

# EfficientRep
# num_repeats=[1, 6, 12, 18, 6]
# out_channels=[64, 128, 256, 512, 1024]
{
[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=silu

}



# type='RepBiFPANNeck'
# num_repeats=[12, 12, 12, 12]
# out_channels=[256, 128, 128, 256, 256, 512]
{
[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=silu

}


# anchors_init=[ [10,13, 19,19, 33,23], 
#                [30,61, 59,59, 59,119]
#                [116,90, 185,185, 373,326]]
{
[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=silu

}