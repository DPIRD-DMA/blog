---
layout: post
title: Train a Deep Learning image classifier in 5 minutes with Python
---
Image classification is the process of assigning a label to an image. This guide will outline how to train a Deep Learning image classifier with a very small amount of code and with limited training data. The approach covered in this post is very powerful and, as such, I find myself using it frequently.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Deep%20Learning%20image%20classifier">link to notebook</a>

Hardware:<br>
To follow this guide, you’re going to need two things, a computer running some form of Linux (I’m using and recommend Ubuntu 20.04) and a recent high-end NVIDIA graphics card (GTX 1060 6GB or better). If you meet these requirements then you are good to go. If not, your best bet is probably to try out Google Colab which offers a free GPU enabled virtual server.

Environment setup:<br>
At this stage, you are going to want to make sure your system is recognising your GPU. Fire up a terminal and type ‘nvidia-smi’. Assuming the GPU drivers are correctly installed, you should see something like this.

<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2021-07-01-Train-a-Deep-Learning-image-classifier-in-5-minutes-with-Python-nvidia-smi.png?raw=true" width="500">

If you are running Ubuntu 20.04, the system should have automatically installed the correct driver, assuming third-party drivers are enabled. If the terminal gives you an error then head over to the NVIDIA site and download and install the latest driver for your particular GPU.
Now that your driver is sorted, you are going to need to install ‘conda’ and then create a new environment with fastai v2 and jupyter notebook installed. The process to do this changes from time to time so head over to the fastai <a href="https://github.com/fastai/fastai">github page</a> to check the latest setup instructions.  

Getting data:<br>
To train your Deep Learning model you will need to supply the model with some manually labelled data. The classic example here is training a model to classify photos of cats and dogs, however, in the last couple of years, this task has become almost too easy. So instead you will train your model to distinguish between ten different breeds of dogs, a significantly harder task. Fortunately, this dataset already exists. It is called ‘Imagewoof’ and you can <a href="https://github.com/fastai/imagenette">download it here</a>, go ahead and download the “320 px download” version and extract the archive. Within the extracted folder, open up the ‘train’ subfolder. Within the ‘train’ folder you will note that each further subfolder contains about one thousand images of one dog breed. There are ten such subfolders, so there are ten dog breeds and about ten thousand photos in total.

Start running the code:<br>
Head over to our repo and download <a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Deep%20Learning%20image%20classifier">this notebook</a>. Start jupyter from the terminal with the command ‘jupyter notebook’. You should now see a new tab in your default web browser, from within this tab, navigate to the downloaded notebook and open it.


Run the first cell, this pulls in the fastai library which contains all the tools you need for this notebook.

{% highlight python %}
from fastai.vision.all import *
{% endhighlight %}

The next cell simply tells the notebook where the data is located. Edit this path to point at the ‘imagewoof2-320/train’ folder that you extracted.
{% highlight python %}
path = Path('/home/nick/Downloads/imagewoof2-320/train')
{% endhighlight %}

You may have noted that the subfolders within ‘train’ are just numbers. This isn’t particularly useful for us, so the next two cells rename the folders to the actual breed names.
{% highlight python %}
label_remap_dict = {'n02086240':'Shih-Tzu',
                'n02087394':'Rhodesian_ridgeback',
                'n02088364':'beagle',
                'n02089973':'English_foxhound',
                'n02093754':'Border_terrier',
                'n02096294':'Australian_terrier',
                'n02099601':'Golden_retriever',
                'n02105641':'Old_English_sheepdog',
                'n02111889':'Samoyed',
                'n02115641':'Dingo'}
{% endhighlight %}

{% highlight python %}
for folder in path.glob('n02*'):
    class_name = label_remap_dict[folder.name]
    folder.rename(path/class_name)
{% endhighlight %}

The next cell contains two variables that may need to be changed depending on your hardware setup. The ‘batch_size’ determines the number of images being processed on your GPU at any point in time. Generally, you want this to be as high as possible, however, if it is too high you may run into GPU/CUDA memory errors and be forced to reset your notebook. Most GPUs should be able to handle a batch size of 10-30 with these 320px images. The other variable here is the image size, this is the other knob you can twist if you are running out of GPU memory. However, if you make it too small the model may not be able to recognise important features in the images. Keep in mind these models are not magic. If you set the image size to the point where you can no longer recognise the thing you are trying to classify, then the model will probably have a hard time too.
{% highlight python %}
batch_size = 30
img_size = 320
{% endhighlight %}

In the next cell, you are defining the Image Augmentations which will be applied to the imagery. Image Augmentations are very useful to artificially increase the quantity of your training data. This is done by slightly modifying the images before the model gets to see them. This significantly improves the accuracy and robustness of models and is critical if you have limited training data. But keep in mind to only apply augmentations that make sense for the data. For instance, flipping the ‘Imagewoof’ images upside down would not be very helpful because the model would probably not need to classify upside-down images. In this particular example, you can use ‘aug_transforms() which has a bunch of presets which have been proven to work well. If you would like to know exactly what this is doing you can type ‘aug_transforms?’ in the new cell, this will show you all the presets for this particular function.
{% highlight python %}
item_tfms = [Resize(img_size)]
batch_tfms = [Normalize.from_stats(*imagenet_stats),
            *aug_transforms(size=img_size)]
{% endhighlight %}

This cell tells fastai how you would like to use the image data. This includes how to label the data (from the subfolder names) as well as how to split the images into a training set and validation set, and finally the Image Augmentations that you would like to use.
{% highlight python %}
data = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2,splitter=RandomSplitter(seed=42),
                                    bs=batch_size,
                                    item_tfms=item_tfms,
                                    batch_tfms=batch_tfms
                                   )
{% endhighlight %}

To make sure everything is working correctly, you can ask fastai to display some images with the corresponding label. Keep in mind these images have had the transforms applied so they will look somewhat different to the raw images in the folders.
{% highlight python %}
data.show_batch(max_n = 9, figsize = (15,15))
{% endhighlight %}

This cell grabs a copy of the image labels and saves them to a .csv file. This step is not necessary to train the model, however, in production, your model will label images with integers that represent your classes, not the actual class names. If you want to convert these integers back to class names you will need this .csv. Also, it’s worth noting you are using NumPy here without importing it explicitly. When you imported fastai using the  ‘import *’ command it automatically imported NumPy for you.
{% highlight python %}
np.savetxt(path/'classes.csv', np.array(data.vocab), fmt='%s')
{% endhighlight %}

Now it’s time to prepare the model. The notebook calls up a ‘cnn_learner’ (Convolutional Neural Network) and is pointed towards your data. It is also directed to use a ‘resnet18’ which is the model architecture and sets the model to mixed-precision mode ‘.to_fp16()’. Mixed-precision tells the model to use 16-bit precision where possible, to reduce memory overhead. Note, mixed-precision mode may not work on older GPUs. If this line is throwing an error try removing ‘.to_fp16()’.
{% highlight python %}
learn = cnn_learner(data, resnet18, metrics=error_rate).to_fp16()
{% endhighlight %}

It is now time to train the model. The first parameter in this function is the ‘epoc’ count, effectively the number of times the model gets to look at each image. Increasing this number will improve the model to a point, however for most image classification problems somewhere between 3-10 is enough to start to see limiting returns. The notebook also has some callbacks here, one to plot a graph of the model’s progress and one that saves the best model to disk. Saving the best model as it trains is quite useful as sometimes while training a model, it will start to go off the rails and you may want a previous generation of the model which performed better.
At this point go grab a coffee, but don’t take too long! Training your model should only take a couple of minutes depending on your hardware. For this particular task, and with the parameters set above, your model should train to around 5-7% error. Keep in mind there is some randomness in the process so you will get some run to run variance.
{% highlight python %}
learn.fine_tune(3,cbs = [ShowGraphCallback(),
                         SaveModelCallback(monitor='error_rate',with_opt=True, fname='quick_model')])
{% endhighlight %}

<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2021-07-01-Train-a-Deep-Learning-image-classifier-in-5-minutes-with-Python-Training.png?raw=true" width="500">

Pretty impressive right? One of the many interesting techniques at work here is using a pre-trained model. Your model did not start off from a blank slate. In the background, fastai downloaded a model which had already been trained on millions of images. All you did was slightly tweak this model to work well for your particular task. Using a pre-trained model is almost always a good idea, even if what you are trying to classify is totally unrelated to what the pre-trained model was trained on.

Now that the model has finished training the notebook, it will reload the best performing version of the model and save it as a .pkl file. This is helpful if you would like to use this model later.
{% highlight python %}
{% endhighlight %}

The next cell loads up some random images from your validation set and displays the true labels along with the model predicted labels.	 Hopefully, most of these are correct!
{% highlight python %}
learn.load('quick_model')
learn.export('models/quick_model.pkl')
{% endhighlight %}

Now you can also load up the top losses. These images are also from the validation set however, these ones are where the model was most wrong. Keep an eye on these as sometimes it can tell you something very interesting about your model.
{% highlight python %}
learn.show_results()
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))
{% endhighlight %}

Lastly, you can plot up a confusion matrix to illustrate how the model is performing on all the image classes. Hopefully, you will see a nice strong diagonal line that indicates a well-behaving model.
{% highlight python %}
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
{% endhighlight %}

<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2021-07-01-Train-a-Deep-Learning-image-classifier-in-5-minutes-with-Python-Confusion_matrix.png?raw=true" width="500">

You are now finished, so maybe try loading up some of your own data and see how the model goes. In the next couple of months, I will write another post showing different ways I use these types of models in production.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Deep%20Learning%20image%20classifier">link to notebook</a>
