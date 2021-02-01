---
layout: post
title: Multispectral image classification with Transfer Learning
---
Heads up, today we are jumping into the deep end of Deep learning with the fastai library. If you haven't spent much time with fastai this walk through may be a little full on.

If you're in the worlds of remote sensing and deep learning, you have no doubt run into the issue of wanting to use Transfer learning but also wanting to use multispectral imagery. Unfortunately there are two major issues when combining these. Firstly, pretrained models (used for Transfer learning) expect that you are going to use RGB imagery and secondly (depending on your library of choice), the built-in image augmentations may also expect RGB imagery. It turns out neither of these issues are showstoppers, they just required a couple days of experimentation and some help from the <a href="https://forums.fast.ai/">fastai forum</a> (specifically Malcolm McLean) to solve.

In the <a href="https://youtu.be/cX30jxMNBUw?t=5065">fastai 2020 Lesson 6 tutorial</a>, Jeremy Howard was asked about using pretrained models for four channel images. Jeremy’s response was that this ‘should be pretty much automatic’. This was exactly what I was after so I went digging and found <a href="https://towardsdatascience.com/how-to-implement-augmentations-for-multispectral-satellite-images-segmentation-using-fastai-v2-and-ea3965736d1">this tutorial</a> by Maurício Cordeiro. This tutorial was very helpful, however I only wanted to do image classification (not segmentation) and I wanted to use pretrained weights for my additional channels, unlike the tutorial which initialised the additional channels with weights of ‘0’, which is the fastai default behavior.

This post will walk through some of the pain-points of multispectral imagery and my work-arounds for dealing with these issues. In particular, this covers creating a custom data loader, modifying a pretrained model and sorting out multispectral augmentations using the fastai (v2) deep learning library for image classification.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Multispectral%20image%20classification%20with%20Transfer%20learning.ipynb">link to notebook</a>

The notebook starts by importing the necessary libraries. The only slightly uncommon libraries here are ‘rasterio’, which is a relatively user friendly interface for ‘GDAL’ raster operations, and ‘albumentations’, which is used for multispectral image augmentations.

{% highlight python %}
import os
from pathlib import Path
import math
import numpy as np
from tqdm.notebook import tqdm
import rasterio
from fastai.vision.all import *
import albumentations as A
{% endhighlight %}

The expected data structure for the notebook is multispectral ‘.tif’ files in folders denoting the class name. Just point the notebook at the parent folder of the data. This cell also adds a folder named ‘models’ which will contain our finished models.

{% highlight python %}
#  this path should contain folders of images of each class
path = Path('/media/nick/2TB Working 3/Projects/Dunes/1000px/multiband/Training data')
# make new folder for models within the training data folder
model_path = os.path.join(path,'models')
Path(model_path).mkdir(parents=True, exist_ok=True)
{% endhighlight %}

The next cell sets the image size; all the images will be resized to a square with this many pixels on each side. This is useful if the data is inconsistently sized or if you just want to down sample your input data to speed up training. The batch size is also set at this point.

{% highlight python %}
# set image size
img_size = 1000
# set batch size
bs = 5
{% endhighlight %}

This cell sets up a bunch of helper functions, mostly for handling tensors and displaying images.

{% highlight python %}
# open an image and convert it to a tensor
def open_img(path):
#     use rasterio to open image as numpy array and rescale to 0-1
#     you may need to change this if you have values above 255
    img_as_array = rasterio.open(path).read().astype('float32')/255.0
#     convert numpy array to tensor
    img_as_tensor = torch.from_numpy(img_as_array)
#     resize tensor if necessary
    img_as_tensor_res = resize_tensor(img_as_tensor)
    return img_as_tensor_res

# resize the dimensions of a tensor
def resize_tensor(input_tensor):
#     from https://stackoverflow.com/questions/59803041/resize-rgb-tensor-pytorch
    tensor_un = input_tensor.unsqueeze(0)
    tensor_res = torch.nn.functional.interpolate(tensor_un,size=(img_size,img_size), mode='bilinear', align_corners=True)
    tensor_sq = tensor_res.squeeze(0)
    return(tensor_sq)

# get the image label from the folder name
def get_label(path):
    label = os.path.basename(os.path.dirname(path))
    return label

# open 3 consecutive channels of a tensor as an image
def show_tensor(tensor, first_channel = 0):
    plt.imshow(tensor[0+first_channel:3+first_channel].permute(1, 2, 0).numpy())

# open image as a tensor then open 3 consecutive channels as an image
def show_tensor_from_path(path, first_channel = 0):
    tensor = open_img(path)
    show_tensor(tensor,first_channel)

# convert 3 consecutive channels of a tensor to numpy array
def tensor_to_np_3b(tensor, first_channel = 0):
    return tensor[0+first_channel:3+first_channel].permute(1, 2, 0).numpy()
{% endhighlight %}

This next cell uses the fastai function ‘get_files()’ to retrieve a list of the training data files. This list will be used later to test the augmentations.

{% highlight python %}
# grab all tif files in 'path'
all_images = get_files(path, extensions='.TIF')
print(len(all_images),'images')
print(all_images[0])
{% endhighlight %}

At this point, the notebook opens a sample of the data to make sure everything is working as expected. The ‘show_tensor()’ function will display three channels of a tensor as an image. In my particular example, I’m dealing with six channel imagery which is actually two stacked RGB images. The notebook is displaying the first three channels as an image and the last three as another image.

{% highlight python %}
# try opening an image and displaying channels 0,1,2 anf 3,4,5
img_num = 2
print(get_label(all_images[img_num]))

input_image = open_img(all_images[img_num])
print(input_image.size())

img = show_tensor(input_image,first_channel=0)
plt.figure()
img = show_tensor(input_image,first_channel=3)
{% endhighlight %}

If the images above look as expected, the data structure is probably correct and you can move on to setting up some augmentations. The built in fastai image augmentations will no longer work as they expect three channel images. To work around this, this notebook uses the ‘albumentations’ library instead. ‘Albumentations’ has implemented augmentations which (mostly) work with multispectral imagery. The list of transforms chosen is roughly based on the default fastai image augmentations as they have been proven to be a good starting point. Keep in mind that these augmentations are performed on your CPU before each epoch, so you may experience a slow down in training if you add many of them.

{% highlight python %}
# we can't use the built in fastai augmentations as they expect 3 channel images so we are using Albumentations instead
# https://github.com/albumentations-team/albumentations
# add as many as you want but these are executed on CPU so can be slow...
transform_list = [A.RandomBrightnessContrast(p=1,brightness_limit=.2),
                  A.RandomRotate90(p=0.5),
                  A.HorizontalFlip(p=.5),
                  A.VerticalFlip(p=.5),
                  A.Blur(p=.1),
                  A.Rotate(p=0.5,limit = 10)
                 ]
{% endhighlight %}

Now that the augmentation list is defined, the notebook sets up a function to apply them. The ‘aug_tfm()’ function intakes a tensor and applies the augmentations one after another. The ‘if’ statement in this function simply checks the length of the input tensor, to avoid augmenting any image labels which are passed to this function. There is probably a better way to deal with this, but this works fine. One other problem here is that some augmentations will shift the tensor values outside the range of 0-1 (such as ‘RandomBrightnessContrast’). To address this, the numpy.clip function is applied to remove any high or low values.

{% highlight python %}
# apply the augmentations in a loop
def aug_tfm(tensor):
#     this function is used for both images and labels so check the count of the input tensor
#     if the count is above 1 its not a label so apply the augmentations
    if tensor.nelement()>1:
#         convert tensor into numpy array and reshape it for Albumentations
        np_array = np.array(tensor.permute(1,2,0))
#        apply each augmentation
        for transform in transform_list:
            np_array = transform(image=np_array)['image']
#         some augmentations may shift the values outside of 0-1 so clip them  
        np_array = np.clip(np_array, 0, 1)
#        rearrange image to tensor format
        array_arange = np_array.transpose(2,0,1)
#        convert back to tensor
        tensor = torch.from_numpy(array_arange)

    return tensor
{% endhighlight %}

Finally, the ‘multi_tfm()’ function is created. This uses ‘RandTransform’ to tell fastai to only apply our augmentations to the training images and not the validation set.

{% highlight python %}
multi_tfm = RandTransform(enc=aug_tfm, p=1)
{% endhighlight %}

Now that all the transforms are sorted, let’s test them out! This next cell grabs an image, opens it as a tensor, and then applies a random set of augmentations six times. Each image you see should be slightly different.

{% highlight python %}
aug_test = open_img(all_images[1])
rows = 2
cols = 4
axes=[]
fig=plt.figure()

for a in range(rows*cols):
    b = tensor_to_np_3b(aug_tfm(aug_test))
    axes.append( fig.add_subplot(rows, cols, a+1) )
    plt.imshow(b)
fig.tight_layout()
fig.set_size_inches(cols*4, rows*4)
plt.show()
{% endhighlight %}

Assuming everything is working, you can now setup the fastai ‘DataBlock’. This cell sets up two blocks; first, a ‘TransformBlock’, which will contain the images, and second, a ‘CategoryBlock’, which will contain the labels. The fastai function ‘get_image_files’ is used to find the paths to all of the images. The ‘get_labels’ function is used to extract the image class from its path. The fastai function ‘RandomSplitter’ is used to split the data into train and validation sets, with a static seed to always get the same split (so you can compare different runs). Lastly, the augmentations are passed in and the ‘DataBlock’ is now completed.

{% highlight python %}
db = DataBlock(blocks=(TransformBlock(open_img), CategoryBlock),
                   get_items = get_image_files,
               get_y= get_label,
               splitter=RandomSplitter(valid_pct=0.2, seed=42),
               item_tfms=multi_tfm,
                             )
dl = db.dataloaders(source=path, bs=bs)
batch = dl.one_batch()
print(batch[0].shape, batch[1])
{% endhighlight %}

The next few cells perform a check. The first one prints the channel count.

{% highlight python %}
channel_count= batch[0].shape[1]
print('Channel count =',channel_count)
{% endhighlight %}

The next one prints one image from the validation image set.

{% highlight python %}
# grab a validation tensor, place it on the CPU then show it, this should not have augmentations
# first image is channels 0,1,2 second is 3,4,5
valid_tensor = dl.valid.one_batch()[0][0].cpu()

show_tensor(valid_tensor, first_channel =0)
plt.figure()
show_tensor(valid_tensor, first_channel =3)
{% endhighlight %}

The last one prints an image from the training set.

{% highlight python %}
# show one tensor from training set with augmentations
# first image is channels 0,1,2 second is 3,4,5
train_tensor = dl.train.one_batch()[0][0].cpu()

show_tensor(train_tensor,first_channel =0)
plt.figure()
show_tensor(train_tensor,first_channel =3)
{% endhighlight %}

The notebook now sets up a learner like normal, however the ’n_in=' variable is set to the channel count, this tells fastai to expect more than three channels.

{% highlight python %}
learn = cnn_learner(dl, resnet18, n_in=channel_count, pretrained=True, metrics=error_rate).to_fp16()
{% endhighlight %}

The extra channels that you just told fastai to expect are not pretrained, all the weights have all been set to a value of ‘0’. You can see this yourself in the fastai <a href="https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py">code here</a>. To get around this, the notebook duplicates the pretrained RGB weights into the newly created input channels. This process is started by getting a reference to the input layer in the cell below.

{% highlight python %}
# grab a reference to the first layer of the model, the layer we need to edit to pull over the pretrained weights
layer1 = learn.model[0][0]
print(layer1)
# access the weights of the layer
l1_weights = layer1.weight
print(l1_weights.shape)
{% endhighlight %}

Now that the notebook has a reference to the first layer, it just duplicates the RGB weights to all the additional channels and then reduces all the weights by the channel count ratio to keep the total value of the input layer the same.

{% highlight python %}
pretrained_channel_count = 3

channel_ratio = channel_count/pretrained_channel_count

# define how many times we need to duplicate the weights
repeat_count = math.ceil(channel_ratio)

# duplicate the RGB weights for all additional channels
#           RGB weights       repeat on 2nd axis     chops off any excess
l1_weights = l1_weights[:,:pretrained_channel_count].repeat(1,repeat_count,1,1)[:,:channel_count]

# rescale weights by channel_ratio
l1_weights = l1_weights / channel_ratio
{% endhighlight %}

That's it! The notebook is now ready to train, just call the learning rate finder and then train as normal!  

{% highlight python %}
learn.lr_find()
{% endhighlight %}


{% highlight python %}
learn.fit_one_cycle(1,1e-2)
{% endhighlight %}

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Multispectral%20image%20classification%20with%20Transfer%20learning.ipynb">link to notebook</a>
