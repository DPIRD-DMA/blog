---
layout: post
title: Histogram Equalization on Sentinel-2 L1C Satellite Images
---

For those wrestling with satellite imagery, clouds can be a real pain. They overexpose scenes, making it tough to get a clear picture of both the atmospheric and terrestrial elements. No worries, though! Today we're tackling this issue with a technique called histogram equalization on Sentinel-2 L1C images. 🌦️

## Why Histogram Equalization?

Imagine trying to look at a bright cloud and a shaded forest in the same image. Standard visualization will wash out one or the other. Histogram equalization adjusts the range of brightness values in your image, giving both clouds and land their moment in the sun, or well, your screen.

## The Code Explained

Let's jump into our Python code. We're using the libraries numpy, rasterio, pathlib and tqdm. 

{% highlight python %}
import numpy as np
import rasterio as rio
from pathlib import Path
from tqdm.auto import tqdm
{% endhighlight %}

### Paths and Bands

We first define the paths to our input and output directories. We also specify the RGB bands we're interested in (4, 3, 2).

{% highlight python %}
s2_l1c_dir = Path('/home/nick/Downloads/test2/in/S2A_MSIL1C...')
output_dir = Path('/home/nick/Downloads/test2/out')
required_bands = [4,3,2]
{% endhighlight %}

### Histogram Functions

We have two core functions: `get_histogram` to calculate the histogram and `histogram_equalization` to apply equalization.

{% highlight python %}
def get_histogram(array, bins, data_pct=1):
    # Calculate the subset size based on the data_pct parameter
    subset_size = int(len(array) * data_pct / 100)
    
    # Randomly choose a subset from the original array
    array = np.random.choice(array, subset_size)
    histogram = np.zeros(bins)
    
    # Count occurrences of each value in the array and populate the histogram
    for value in array:
        histogram[value] += 1
    return histogram

def histogram_equalization(array):
    # Flatten the array to a 1D array
    flat = array.flatten()
    
    # Get the histogram of the flattened array
    hist = get_histogram(flat, 65536)
    
    # Calculate the cumulative sum of the histogram
    cs = np.cumsum(hist)
    
    # Normalize the cumulative sum
    cs = ((cs - cs.min()) * 65535 / (cs.max() - cs.min())).astype('uint16')
    
    # Reshape the normalized values back to the original array shape
    return np.reshape(cs[flat], array.shape)
{% endhighlight %}

### Looping Through Bands

Then we loop through each band, applying histogram equalization. Because clouds aren't going to equalize themselves, you know.

{% highlight python %}
rgb_stack = []

for band in tqdm(required_bands):
    # find band path
    band_path = list(s2_l1c_dir.rglob(f'*GRANULE/*/IMG_DATA/*_B0{band}.jp2'))[0]
    # open array
    src = rio.open(band_path)
    array = src.read(1)
    # equalize array
    array = histogram_equalization(array)
    # convert to uint8
    array = (array / 256).astype('uint8')
    rgb_stack.append(array)
{% endhighlight %}

### Why uint8 and JPEG Compression?

We convert the equalized array to `uint8` to save space. It still maintains good enough quality for visualization. We then use JPEG compression in our GeoTIFF to save even more space.

{% highlight python %}
# stack bands into 3D array
rgb_stack = np.array(rgb_stack)
# copy and update band metadata
meta = src.meta.copy()
meta.update({
    'count': rgb_stack.shape[0],
    'dtype': rgb_stack.dtype,
    'compress': 'JPEG',
    'driver': 'GTiff'
})
{% endhighlight %}

### Finale: Writing the Image

Finally, we write the image into a GeoTIFF file.

{% highlight python %}
# export equalized compressed RGB image
with rio.open(output_dir / f'{s2_l1c_dir.stem}_RGB.tif', 'w', **meta) as dst:
    dst.write(rgb_stack)
{% endhighlight %}

## Before and After Screenshots

Before Equalization             |  After Equalization
:-------------------------:|:-------------------------:
!<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2023-09-20-Histogram-Equalization-on-Sentinel-2-L1C-Satellite-Images_1.png?raw=true" width="1850">
  |  !<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2023-09-20-Histogram-Equalization-on-Sentinel-2-L1C-Satellite-Images_2.png?raw=true" width="1850">


Notice how you can see both clouds and land features clearly in the equalized image. So the next time clouds crash your image party, you know how to make them behave!


That's all folks, happy equalizing! 🌈🛰️

### Acknowledgements
Some of the ideas and techniques discussed in this blog post were inspired by Tory Walker's GitHub repository on histogram equalization, which you can find [here](https://github.com/torywalker/histogram-equalizer/blob/master/HistogramEqualization.ipynb).
