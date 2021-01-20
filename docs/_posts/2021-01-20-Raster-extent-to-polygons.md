---
layout: post
title: Raster extent to polygons
---
Recently, I had the need to visualise the extent of a large number of DEM files. I initially tried loading them into QGIS but this was very clunky and slow. I only wanted to see the extent of the files so I attempted to use the inbuilt QGIS function ‘Extract layer extent’ and run it as a batch process. Unfortunately the batch processing window did not appreciate me trying to load a couple hundred large DEM files and it promptly crashed. So I put a Jupyter Notebook together to do the work for me. This notebook will crawl all files within a directory and all subdirectories and extract the bounding geometry for each file. These bounds are then grouped by projection and saved out as a geopackage ready to be viewed in QGIS or alike.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Raster%20extent%20to%20polygon.ipynb">link to notebook</a>


The notebook starts by importing the necessary libraries, ‘os’ for file path manipulation and finding the rasters, ‘rasterio’ for extracting the raster bounds, ‘shapely’ for creating polygon, ‘pandas’ and ‘geopandas’ for dealing with collections of polygons, ‘multiprocessing’ for speeding things up and ‘tqdm’ for displaying loading bars.

{% highlight python %}
import os
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from multiprocessing import Pool
from tqdm import tqdm
{% endhighlight %}

The notebook now requires you to input the path to the folder which contains the raster files as well as the file extension you are looking for. The export directory and file names are automatically derived from the input directory.

{% highlight python %}
input_dir = '/home/nick/Desktop/test'
file_types = ('.ers','.tif')
export_dir = os.path.dirname(input_dir)
export_file_name = os.path.basename(input_dir)
{% endhighlight %}

Now the notebook will pass the ‘input_dir’ to os.walk which will search your directory for all files with the extensions specified above. The result of this will be pushed into the ‘raster_list’.  

{% highlight python %}
raster_list = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(file_types):
            raster_list.append(os.path.join(root, file))

len(raster_list)
{% endhighlight %}

The next step is to define a function which will try to open each raster and extract the crs and bounds of each file. ‘Rasterio’ is used to open the raster file, extract the crs and then the bounds. The bounds themselves are not very useful, it just expresses the ‘left’, ’bottom’, ’right’ and ‘top’ of the file. Fortunately we can pass this to the shapely function ‘box’ which turns the bounds into a polygon which geopandas will recognise. The function then returns a dictionary of the relevant info. If any of these steps fail for whatever reason the function will print out the path to the offending file.


{% highlight python %}
def index(raster):
    try:
        dataset = rasterio.open(raster)
        proj = dataset.crs.to_string()
        bounds = dataset.bounds
        geom = box(*bounds)
        raster_name = os.path.basename(raster)
        return({'name':raster_name,'path':raster,'crs':proj,'geometry':geom})
    except:
        print('could not open',raster)
{% endhighlight %}



The notebook now uses ‘Pool()’ to run the above function with multiple processes. Within this, ‘tqdm’ is used to display a loading bar, the result is a list of dictionaries within variable the ‘polygons’.

{% highlight python %}
with Pool() as p:
    polygons = list(tqdm(p.imap(index, raster_list), total=len(raster_list)))
{% endhighlight %}


The next cell converts this list into a pandas dataframe and extracts a list of all the unique crs identifiers. This list is then printed out as well as the head of the dataframe so you have a preview of your data.

{% highlight python %}
all_polygons = pd.DataFrame(polygons)
crs_list = all_polygons['crs'].unique()
print(crs_list)
all_polygons.head()
{% endhighlight %}

The notebook now exports your data, however it’s possible you will have multiple different crs identifiers, this will cause a problem because you can’t have multiple identifiers in the same file. To resolve this, the notebook loops over each unique crs and only exports the relevant extents for the crs. This crs is also embedded within the name of the exported file which is also printed out.

{% highlight python %}
for crs in tqdm(crs_list):
    one_crs_df = all_polygons[all_polygons['crs']==crs]
    one_crs_gdf = gpd.GeoDataFrame(one_crs_df,crs=crs) #crs={'init' :crs}
    save_name = os.path.join(export_dir,(export_file_name+" crs "+crs.split(':')[1]+'.gpkg'))
    one_crs_gdf.to_file(save_name, driver ="GPKG")
    print(save_name)
{% endhighlight %}

You should now have one or more .gpkg files within the directory above your input directory and be able to load these into your GIS viewing program of choice.

Note that python does not do a good job of memory management, so keep an eye on your RAM usage if you are processing large files. If you manage to run out of free RAM, the notebook will probably crash and may lock up your computer for a while. To avoid this happening you can limit the amount of files which are allowed to be open at the same time by placing a number (smaller than your computer’s number of threads) within the brackets of ‘Pool()’. This will limit the number of simultaneous processes, this number defaults to your computer’s thread count.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Raster%20extent%20to%20polygon.ipynb">link to notebook</a>
