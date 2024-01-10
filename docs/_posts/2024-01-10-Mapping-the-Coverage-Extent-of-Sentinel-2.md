---
layout: post
title: Mapping the Coverage Extent of Sentinel-2
---
## Introduction

The Sentinel-2 satellite constellation plays a pivotal role in remote sensing and data science, especially in monitoring terrestrial environments. However, it faces challenges with inconsistent data coverage over oceanic regions, particularly at the edges of its coverage. This inconsistency poses difficulties for marine and coastal research, particularly in time series analysis. To mitigate this issue, we've developed a Python tool that leverages the Microsoft Planetary Computer API. This tool calculates and visualises the global capture extent of Sentinel-2 by processing and amalgamating extent polygons into a global raster. The resulting raster provides a pixel-wise representation of observation frequency, aiding researchers and analysts in navigating the challenges of data coverage irregularities.

Inconsistent Sentinel-2 scene extents 
<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2024-01-10-Mapping-the-Coverage-Extent-of-Sentinel-2_image_1.png?raw=true" width="1850">
Sentinel-2 coverage heatmap
<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2024-01-10-Mapping-the-Coverage-Extent-of-Sentinel-2_image_2.png?raw=true" width="1850">


### Overview

Our tool utilises the Python libraries geopandas, rasterio, and pystac to efficiently handle geospatial data. The methodology involves querying the Planetary Computer API for Sentinel-2 scenes over a designated timeframe and merging the polygon extents of each scene's valid data to form a global raster. This raster illustrates the coverage frequency at each pixel location. Given the substantial volume of API requests, we execute these in parallel. The final output is an exported geotiff, offering a detailed visualisation of Sentinel-2 coverage frequency.

Global Sentinel-2 coverage heatmap for 2023
<img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2024-01-10-Mapping-the-Coverage-Extent-of-Sentinel-2_image_3.png?raw=true" width="1850">

### Installation and Usage

Setting up the tool is simple. Ensure Python is installed, then install the required dependencies via pip. The tool can then be cloned and run locally or on the Planetary Computer Hub, with optional arguments for start and end years, and the desired output raster resolution.

Basic usage example:
{% highlight python %}

from helpers.coordinator import build_revisit_raster

raster_path = build_revisit_raster()

{% endhighlight %}

### Try It Out

Explore the tool on GitHub: [Sentinel-2 Capture Frequency Tool on GitHub](https://github.com/DPIRD-DMA/Sentinel-2-capture-frequency).


### Acknowledgements
Thanks to Github user [justinelliotmeyers](https://github.com/justinelliotmeyers) for providing the Sentinel-2 [scene extents](https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index) and to the [Planetary Computer](https://planetarycomputer.microsoft.com/) for providing easy and rapid access to the Sentinel-2 archive.
