---
layout: post
title: "S2Mosaic: Crafting Cloud-Free Sentinel-2 Mosaics with Ease"
---

Clouds in satellite imagery: the persistent nemesis of remote sensing analysts everywhere. They obscure our view, skew our data, and generally make life difficult. But fear not! Today, we're introducing S2Mosaic, a Python package designed to create cloud-free mosaics from Sentinel-2 imagery. 🛰️☁️🌈

## Why S2Mosaic?

Picture this: you're trying to analyze water extent with S2 data, but half your images are covered in clouds. Frustrating, right? S2Mosaic resolves this by intelligently selecting and combining multiple Sentinel-2 scenes to create a single, truly cloud-free composite!

## The Code Explained

Let's dive into it with a simple example.

{% highlight python %}
from s2mosaic import mosaic
from pathlib import Path
{% endhighlight %}

### Setting Up Parameters

First, we define our mosaic parameters. We're targeting a specific S2 grid area and time range, and we want a visual (true color) composite.

{% highlight python %}
grid_id = "50HMH"
start_year = 2022
start_month = 1
start_day = 1
duration_months = 2
output_dir = Path("output")
required_bands = ['visual']
{% endhighlight %}

### Creating the Mosaic

Now for the magic! We call the `mosaic` function with our parameters:

{% highlight python %}
result = mosaic(
    grid_id=grid_id,
    start_year=start_year,
    start_month=start_month,
    start_day=start_day,
    duration_months=duration_months,
    output_dir=output_dir,
    required_bands=required_bands,

)

print(f"Mosaic saved to: {result}")
{% endhighlight %}

### What's Happening Under the Hood?

1. **Scene Selection**: S2Mosaic searches for all available Sentinel-2 scenes within our specified time range and grid id.
2. **Sorting**: Scenes are sorted based on the percentage of valid (cloud-free) pixels (or image date if specified).
3. **Cloud Masking**: Each scene is processed using the state-of-the-art OmniCloudMask to identify and mask out clouds and shadows.
4. **Compositing**: The selected scenes are combined using the mean of valid pixels (or first or last valid pixel if specified) to create a single, cloud-free mosaic.
5. **Output**: The final mosaic is saved as a GeoTIFF file in our specified output directory (or returned as an array if no output directory is supplied).

## Advanced Usage: Customizing Your Mosaic

S2Mosaic is flexible! Let's look at a more advanced example where we specify particular bands and tweak some parameters:

{% highlight python %}
array, profile = mosaic(
    grid_id="50HMH",
    start_year=2022,
    start_month=1,
    start_day=1,
    duration_months=2,
    sort_method="oldest",
    mosaic_method="first",
    required_bands=["B04", "B03", "B02", "B08"],
    no_data_threshold=0.001,
    ocm_batch_size=4,
    ocm_inference_dtype="bf16"
)

print(f"Mosaic array shape: {array.shape}")
print(profile)
{% endhighlight %}

In this case, we're:
- Selecting specific bands (Red, Green, Blue, and Near Infrared)
- Adjusting the cloud masking parameters for optimal performance
- Getting the mosaic as a NumPy array and rasterio profile instead of saving to disk

## Example Output

Below is an example output with images taken from the South coast of Western Australia, a notoriously cloudy area. The imagery dates from 01/06/2024-01/08/2024.

The settings used to create this output
{% highlight python %}
mosaic(grid_id='50HNG',
        start_year=2024,
        start_month=6,
        duration_months=2,
        required_bands=['visual'],
        no_data_threshold=0.001,
        )
{% endhighlight %}

<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2024-08-01-S2Mosaic-Creating-Cloud-Free-Sentinel-2-Mosaics_image_1.png?raw=true" width="1850">

Notice how the cloudy patchwork has been transformed into a clear, continuous view of the landscape.

That's all folks! The next time clouds try to rain on your remote sensing parade, you'll have S2Mosaic in your toolkit to clear things up. Happy mosaicking! 🌍🛰️🧩

### Acknowledgements
A big thanks to the Microsoft Planetary Computer for making the data easily accessible and the entire open-source remote sensing community for their continuous innovations.