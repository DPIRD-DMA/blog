---
layout: post
title: Point sampling multiple raster files
---
When performing spatial modeling it is often necessary to extract raster values at xy point locations. If you're working with small to moderate amounts of data, this operation can be done within QGIS or alike, however, if you are working with larger datasets or many small datasets it becomes useful to use a script to do the work for you. I ran into this issue recently when I needed to extract 150,000 point values from several hundred raster files. Needless to say, QGIS did not appreciate me trying to load all of this data into it, so I resorted to building a jupyter notebook instead.

This notebook intakes a point file and a folder of rasters and will efficiently extract the raster values at the specified point locations.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Point%20sampling%20multiple%20raster%20files.ipynb">link to notebook</a>

The notebook starts by importing the necessary libraries,  using ‘os’ for file path manipulation, ‘multiprocessing’ for iterating over multiple rasters at once, ‘tqdm’ for displaying a loading bar, ‘rasterio’ for extracting the raster values and finally ‘geopandas’ for importing, displaying and exporting the point files.

{% highlight python %}
# system
import os
from multiprocessing import Pool
from tqdm.auto import tqdm
# GIS
import rasterio
import geopandas as gpd
{% endhighlight %}

Next, the input point file, export file name and raster folder must be defined. Geopandas is pretty flexible and should be able to load just about any vector file format. The export file will be placed in the same folder as the input file so make sure the names are not the same or you may override your input data. The raster folder should contain the raster files you would like to sample, the notebook will scan this folder and all sub folders for raster files ending with the specified file extensions.

{% highlight python %}
# input data can be gpkg shp ect
input_point_folder = '/media/nick/test'
input_point_file_name = 'point_data.gpkg'

# the name of the export file
export_point_file_name = 'points with raster values.gpkg'

# the folder with raster files
raster_folder = '/home/nick/Downloads/test'
# the raster file types
raster_file_types = ('.tif','.tiff')
{% endhighlight %}

At this point the notebook will try to load the point data. It’s worth noting that to intersect the point data with the raster data, they must have the same crs. If this is not the case it’s almost always more efficient to reproject the point data rather than the raster data. To do this, change the ‘EPSG’ code in this cell to match that of the raster data. If all your data is in the same projection you can just comment out this line.

{% highlight python %}
# make input path
input_path = os.path.join(input_point_folder,input_point_file_name)
# open point data
points = gpd.read_file(input_path)
# reproject point data if needed, or comment it out
points = points.to_crs("EPSG:4326")
# display the head of the point values
points.head()
{% endhighlight %}

It is often useful to visualise data to make sure nothing has gone wrong. Geopandas has a ‘plot’ function which shows the entire geodataframe, make sure this appears as you expect.

{% highlight python %}
#display points
points.plot(marker='*', color='green', markersize=5,figsize = (10,10));
{% endhighlight %}

The next cell will strip out the x and y coordinates from the geodataframe and compile them into a list of tuples. To do this, both x and y are extracted as lists then combined with the ‘zip’ function. The first 10 tuples are displayed to make sure this worked as expected.

{% highlight python %}
# make list of xy points from the geodataframe
x = points['geometry'].x.tolist()
y = points['geometry'].y.tolist()
xy_list = list(zip(x,y))
xy_list[:10]
{% endhighlight %}

Now that the xy data is formatted as needed, the notebook searches for the raster files. The ‘os.walk’ function is used in a loop to recursively search subfolders for all files ending with the predefined file extensions. All the files found are appended to the ‘raster_list’.

{% highlight python %}
# make a list of all the raster files
raster_list = []
for root, dirs, files in os.walk(raster_folder):
    for file in files:
        if file.endswith(raster_file_types):
            raster_list.append(os.path.join(root, file))
print('items in list',len(raster_list))
{% endhighlight %}

The notebook now defines a function to extract the raster data. Within this function, rasterio is used to open the raster and the ‘sample’ module is called on that open data to extract all the raster values at the predefined xy locations. This function finishes by returning the name of the raster file and the extracted values as a dictionary.

{% highlight python %}
def point_samp(raster):
    # get raster file name
    raster_name = os.path.basename(raster)
    # open the raster file with rasterio
    src = rasterio.open(raster)
    # use 'sample' to extract the raster values at the xy locations
    raster_vals = [item[0] for item in src.sample(xy_list)]
    # return a dict of the raster name and values for each raster
    return {'name':raster_name,'list':raster_vals}
{% endhighlight %}

At this point the ‘point_samp’ function could be called in a for loop to extract values one at a time from the raster path list. However this would be pretty slow, so instead the notebook uses ‘Pool’ from the multiprocessing library to extract values from multiple rasters at the same time. This multiprocessing is done from within tqdm to also produce a loading bar.

{% highlight python %}
# extract data with multible processes
raster_values = list(tqdm(Pool().imap(point_samp, raster_list), total=len(raster_list)))

# to extract points without multiprocessing use this instead
# raster_values = []
# for raster in tqdm(raster_list):
#     raster_values.append(point_samp(raster))
{% endhighlight %}

Now that all the data has been extracted, the notebook combines all the outputs back together into the initial ‘points’ geodataframe with the raster filename as the column heading.

{% highlight python %}
# add the raster values back in to the geodataframe
for col in raster_values:
    points[col['name']] = col['list']
points.head()
{% endhighlight %}

Assuming the data above looks as expected, these next few cells export out the dataframe as a ‘.gpkg’ geopackage and then a ‘.csv’ comma-separated values.

{% highlight python %}
# make export path for a geopackage
export_gpkg_path = os.path.join(input_point_folder,export_point_file_name)
export_gpkg_path
{% endhighlight %}

{% highlight python %}
# export geopackage
points.to_file(export_gpkg_path, driver='GPKG')
{% endhighlight %}

{% highlight python %}
# make export path for csv based on geopackage path
export_csv_path = export_gpkg_path.replace('.gpkg','.csv')
export_csv_path
{% endhighlight %}

{% highlight python %}

# export csv
points.to_csv(export_csv_path, index=False)
{% endhighlight %}

You are now finished and should be able to load either of the exported files into your GIS viewing software of choice to check your results. 

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Point%20sampling%20multiple%20raster%20files.ipynb">link to notebook</a>
