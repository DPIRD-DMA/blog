---
layout: post
title: Detection of corrupt raster files with python
---
Have you ever needed to find a corrupt GeoTIFF file amongst a large amount of valid files? I recently had this issue and put together a useful little python script to do the work for me. The script is written within a jupyter notebook so you can run it interactively.

<a class="jn" href="https://github.com/DPIRD-DMA">link to notebook</a>

Start by importing the “os” module for its ability to join file paths, search for files and remove files. Import “Pool” from the “multiprocessing” library for parallelisation. The “tqdm” module is imported to show a loading bar to indicate progress, lastly import ‘rasterio’ to open and validate the raster files.
{% highlight python %}
import os
from multiprocessing import Pool
from tqdm.auto import tqdm
import rasterio
{% endhighlight %}
<!-- ![_config.yml]({{ site.baseurl }}/images/image3.png) -->

The second cell defines the formats of interest as a tuple and the input directory which contains the raster files. Keep in mind this script will search all subdirectories of this path as well, which is really handy if you have a nested directory structure.
{% highlight python %}
formats = ('.tif','.tiff','.jpg','.png')

input_folder = '/media/nick/2TB Working 4/Dataset/VIVID 10m'
{% endhighlight %}

<!-- ![_config.yml]({{ site.baseurl }}/images/image4.png) -->

Set up an empty array. This will be filled using the os.walk method, which will search the directory and return every file which ends with one of the predefined formats. It will add each file path to the array “raster_list”. When this has finished it will print out the total amount of raster files found.
{% highlight python %}
raster_list = []

for path in os.walk(input_folder):
    root, dirs, files = path
    for file in files:
        if file.endswith(formats):
            raster_list.append(os.path.join(root, file))

print('Raster count',len(raster_list))
{% endhighlight %}
<!-- ![_config.yml]({{ site.baseurl }}/images/image5.png) -->

The function below will open a raster and check if it is valid. The function takes a raster path as input and tries to open it with rasterio within a try block. This block will fail if rasterio can not open or read the data from the path. Within the try block, rasterio also attempts to extract the bounds (bounding geometry) of the raster file. If “0” is returned for its left coordinate it’s safe to assume (in most cases) the raster coordinate data is missing or broken. If rasterio cant open the data or the coordinates appear to be missformed we pass the raster file path with the return function. If the raster passes both of these tests it will automatically return “None”.

This may not be a fool proof solution but it has worked for me. If you are having specific problems with your files that this does not catch, you can always add more tests.
{% highlight python %}
def img_test(tif):
    try:
        src = rasterio.open(tif)
        read_as_array = src.read()
        bounds = src.bounds
        if bounds.left == 0:
            return(tif)

    except:
        return tif
{% endhighlight %}
<!-- ![_config.yml]({{ site.baseurl }}/images/image2.png) -->

At this stage, a simple for loop could be used to iterate over each raster with the above function, but instead use multiprocessing which will dramatically speed up the process. Using “Pool().imap()” will break the task up into smaller chunks and process multiple chunks at the same time. This process is wrapped within “tqdm()” which provides a loading bar below the cell to show the current progress.

The result of this operation is a list which contains the paths to all the corrupted files as well as a bunch of “None” values which correlates to all the files which passed the test.
{% highlight python %}
currupt_images = list(tqdm(Pool().imap(img_test, raster_list), total=len(raster_list)))
{% endhighlight %}
<!-- ![_config.yml]({{ site.baseurl }}/images/image1.png) -->

To print out all the corrupted files which failed the test, the below loop prints out everything that was returned that is not “None”, these are your corrupted files.
{% highlight python %}
for img in currupt_images:
    if img != None:
        print(img)
{% endhighlight %}
<!-- ![_config.yml]({{ site.baseurl }}/images/image7.png) -->

Finally the last bit of code will delete all the files which failed the test.

<!-- ![_config.yml]({{ site.baseurl }}/images/image6.png)   -->

{% highlight python %}
for img in currupt_images:
    if img != None:
        os.remove(img)
{% endhighlight %}


<a class="jn" href="https://github.com/DPIRD-DMA">link to notebook</a>
