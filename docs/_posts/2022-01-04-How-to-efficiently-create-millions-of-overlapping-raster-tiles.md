---
layout: post
title: How to efficiently create millions of overlapping raster tiles
---
When dealing with large amounts of raster spatial data, you will often find that operations become very slow to perform or just won’t run at all. Often this is the result of single-core processes or simply running out of RAM. Fortunately, in most situations, there is a solution to this issue. Simply chop your raster data into smaller parts and run multiple simultaneous operations. In this post, I will be covering the first half of this workflow; chopping up your data AKA ‘tiling’.

This script is one of a handful of tools I use for preprocessing raster data for Deep Learning models, however, it works just as well for other raster datasets such as DEMs. This script expects your input data to be in the geotiff format with the extension ‘.tif’. However, it should handle any raster that GDAL will open. It also expects that you either have one or more input rasters, if you have multiple input rasters they should be edge matched and have the same resolution and projection. If they do not, you may get unexpected results.

It is worth noting that I have only run this code on Linux and macOS. I believe the multiprocess library (which this script uses extensively) operates slightly differently in different operating systems (beware Windows users).

Also thanks to Carlos Babativa Rodriguez for the help with this one.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/How%20to%20efficiently%20create%20millions%20of%20overlapping%20raster%20tiles.ipynb">link to notebook</a>

The notebook starts by importing the necessary libraries. You will need to have all of these installed within your environment to run this notebook. In my particular case, I’m using Anaconda, so if my environment was lacking ‘tqdm’, I would head over to <a href="https://anaconda.org/conda-forge/tqdm">https://anaconda.org/conda-forge/tqdm</a> and run the first install line on that page.

{% highlight shell_session %}
conda install -c conda-forge tqdm
{% endhighlight %}

Keep in mind you should avoid mixing conda channels, so either stick to the default channel for all your third party installs or stick to the ‘forge’ channel. If you mix channels you may end up breaking your environment, ask me how I know...

{% highlight python %}
import os
import subprocess
from pathlib import Path

import math
from pathlib import Path
from tqdm.auto import tqdm
from multiprocess import Pool,cpu_count
%matplotlib inline

from osgeo import gdal,osr
import geopandas as gpd
from shapely import geometry
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
{% endhighlight %}

This next cell requires you to set some values. Firstly set the desired x and y pixel counts of your tiles as well as the desired overlap. All of these values should be entered as integers as you can’t create part of a pixel. Next, you must enter the folder which contains your rasters, this folder may have a flat structure or be a nested directory, either will work. The output folder will be the location that contains the output tiles. If you define a location that does not yet exist it will be created later. If you only wish to use a subset of your rasters in your directory, you can also define a path to a .csv file. This file should contain a list of raster filenames that you want to use. If you want to use all of your raster files, set this to an empty string. You will also need to define the output raster format, most of the tile ‘.tif’ should do the job. Lastly, the output compression must be defined. If you are using RGB imagery you should probably set ‘JPEG’ assuming you can deal with lossy compression. If you are using almost any other raster data type, ‘LZW’ should do a good job.

{% highlight python %}
tile_size_px = [500,500] #x,y
tile_oxerlap_px = 150

geotiff_folder = '/data/Road extraction/test area 10/test area 10'
output_folder = '/data/Road extraction/test area 10/tiles'
#if you want to use all of your rasters set the below value to ''
valid_rasters_csv = ''

input_file_ext = '.tif'
output_compression = 'JPEG'  #use JPEG for images and LZW for raw data
{% endhighlight %}

This cell allows you to set the output folder structure and tile filename.

{% highlight python %}
# Do you want the output tiles to be nested into row folders?
output_to_row_folders = True
# do you want to append a string to every tile?
raster_name_append = ''
{% endhighlight %}

The settings can now be written out as a ‘json’ file, this is helpful if you forget the specific settings used to create a dataset.

{% highlight python %}
# dump tile setting a a json file into output location
parent_dir = os.path.dirname(output_folder)
settings_file_path = os.path.join(parent_dir,'tiling_settings.json')
settings = {'tile_size_px':tile_size_px,
           'tile_oxerlap_px':tile_oxerlap_px,
           'raster_name_append':raster_name_append,
           'tile_format':'.tif'}
with open(settings_file_path, 'w') as fp:
    json.dump(settings, fp)
{% endhighlight %}

Now that the input raster folder has been validated, this cell will look through it and return all rasters with the ‘input_file_ext’ extension.

{% highlight python %}
Path(output_folder).mkdir(parents=True, exist_ok=True)
os.path.isdir(geotiff_folder)
{% endhighlight %}

If you have set ‘valid_rasters_csv’ to an empty string, the next three cells will not run. Sometimes you may not wish to tile all of your input raster files, in which case you can subset this list by providing a ‘valid_rasters_csv’. It is expected that this file contains a list of filenames which you would like to keep. Any raster that is not in this list will be removed from the input raster list and tiles will not be made from it.

{% highlight python %}
# search folder and all sub folders for 'input_file_ext' files
geo_tiff_list = []
for root, dirs, files in os.walk(geotiff_folder):
    for file in files:
        if file.endswith(input_file_ext):
            geo_tiff_list.append(os.path.join(root, file))

print('We found ',len(geo_tiff_list),input_file_ext,'files')
{% endhighlight %}

This cell will open the ‘valid_rasters_csv’ file with pandas and print the start of the pandas dataframe. Note the column heading of the filenames, in the example; ‘raster_name’.

{% highlight python %}
# if you want to filter our input raster use this
if valid_rasters_csv:
    valid_raster_df = pd.read_csv(valid_rasters_csv)
    print(valid_raster_df.head())
{% endhighlight %}

This cell converts a dataframe column into a set. Note that you will need to replace ‘raster_name’ with your column heading. A set is used here as you should not have any repeating values and because the next cell performs many ‘in’ checks which happens very quickly on sets and much slower on lists.

{% highlight python %}
# convert the valid raster names into a set so the lookup if much faster
if valid_rasters_csv:
    valid_raster_names = set(valid_raster_df.name.tolist())
    valid_raster_names
{% endhighlight %}

Now that the ‘valid_raster_names’ is created, this next cell will loop over each raster and check to see if each raster is in the ‘valid_raster_names’ set. If they are not in the set, they are removed.

{% highlight python %}
if valid_rasters_csv:
    short_list = []
    for tif in tqdm(geo_tiff_list):
    #     my csv file does not have the file extension so we are stripping out the '.tif' to find a match
        file_name = os.path.basename(tif)#.replace('.tif','')
    #     check if the current raster is in the list, if so add it to the clean list
        if file_name in valid_raster_names:
            short_list.append(tif)
    # reset the full list to the clean list        
    geo_tiff_list = short_list
    print(len(geo_tiff_list))
{% endhighlight %}

The following cell defines a function which, opens a given raster and extracts its bounds, returning the bounds and the raster path. This is defined as a function as it will be called in the next cell using multiprocessing.

{% highlight python %}
def get_bounds(tif_path):
#     open file
    data = gdal.Open(tif_path)
#     grab bounds
    geoTransform = data.GetGeoTransform()
    left = geoTransform[0]
    top = geoTransform[3]
    right = left + geoTransform[1] * data.RasterXSize
    bottom = top + geoTransform[5] * data.RasterYSize
#     build dict to file bounds
    geo_tiff_bounds_dict = {'top':top,'left':left,'bottom':bottom,'right':right,'tif_path':tif_path}
    return geo_tiff_bounds_dict
{% endhighlight %}

Note, here we are calling the ‘get_bounds’ function with multiprocessing. This allows as many copies of this function to run as your computer has threads simultaneously. In addition, a loading bar is shown of the current process via ‘tqdm’

{% highlight python %}
# use multiprocessing to extract raster bounds
with Pool() as pool:
    geo_tiff_bounds = list(tqdm(pool.imap(get_bounds, geo_tiff_list), total=len(geo_tiff_list)))
{% endhighlight %}

Now that the bounds of every raster are known, this next cell extracts the maximum geographical extent in each direction, defining one bounding box for all of the input rasters.

{% highlight python %}
# make new array with only bounds to extract full raster extents
pure_bounds = []
for geo_tif_bounds in geo_tiff_bounds:
    pure_bounds.append([geo_tif_bounds['top'],geo_tif_bounds['left'],geo_tif_bounds['bottom'],geo_tif_bounds['right']])
# convert into numpy array
pure_bounds_np = np.array(pure_bounds)
# grab max extents
bound_y_max = float(pure_bounds_np[:,0].max()) #top
bound_x_min = float(pure_bounds_np[:,1].min()) #left
bound_y_min = float(pure_bounds_np[:,2].min()) #bottom
bound_x_max = float(pure_bounds_np[:,3].max()) #right
{% endhighlight %}

To know how to cut the tiles, it is necessary to know the pixel size of the input rasters. As we are assuming all the rasters are the same pixel resolution, this cell opens up the first raster in the list and extracts the pixel size.

{% highlight python %}
# open one image to get the pixel size, this is necessary to know how to cut the rasters
test_raster = gdal.Open(geo_tiff_list[0])
test_raster_gt =test_raster.GetGeoTransform()
pixel_size_x = test_raster_gt[1]
pixel_size_y = test_raster_gt[5]
print(pixel_size_x,pixel_size_y)
{% endhighlight %}

While we are at it, the next cell extracts the projection.

{% highlight python %}
proj = osr.SpatialReference(wkt=test_raster.GetProjection())
crs = 'EPSG:'+proj.GetAttrValue('AUTHORITY',1)
crs
{% endhighlight %}

Using the values supplied and derived above, this next cell calculates the x and y distance between each tile and the size of each tile. Note that if you are creating any overlap, these values will be different from each other.

{% highlight python %}
# calculate the geographical distance in each direction each tile must be from the last tile
x_move = pixel_size_x*(tile_size_px[0]-tile_oxerlap_px)
y_move = pixel_size_y*(tile_size_px[1]-tile_oxerlap_px)
print(x_move,y_move)

# calculate the geographical size of each tile
x_tile_size = pixel_size_x*tile_size_px[0]
y_tile_size = pixel_size_y*tile_size_px[1]
print(x_tile_size,y_tile_size)
{% endhighlight %}

Now that the above has been calculated, the following cells can work out the number of rows and columns that will be tiled. This is needed to cut the tiles in a multiprocessing manner.

{% highlight python %}
# calculate the number of cols so we can avoid using while loops
number_of_cols = math.ceil(abs((bound_x_max-bound_x_min)/x_move))
print(number_of_cols)
# calculate the number of rows so we can avoid using while loops
number_of_rows = math.ceil(abs((bound_y_max-bound_y_min)/y_move))
number_of_rows
{% endhighlight %}

This next cell defines a function that checks which tiles intersect which input rasters. This function will reduce the complexity of the next cell.

{% highlight python %}
# will return a list of geotiffs which intersect
def intersect_tile_with_geotiffs(tile_dict,geo_tiff_bounds):
#     setup set to collect results in, a set is used to avoid duplicates
    intersecting_geotiffs = set()
#     loop over each geotiff
    for geo_bounds in geo_tiff_bounds:
#         check is tile top or bottom is inside geotiff
        if (geo_bounds['top'] > tile_dict['top'] > geo_bounds['bottom'] or
            geo_bounds['top'] > tile_dict['bottom'] > geo_bounds['bottom']):
#         check if left or right are inside a geotiff
            if geo_bounds['right'] > tile_dict['left'] > geo_bounds['left']:
                intersecting_geotiffs.add(geo_bounds['tif_path'])
            if geo_bounds['right'] > tile_dict['right'] > geo_bounds['left']:
                intersecting_geotiffs.add(geo_bounds['tif_path'])
    return intersecting_geotiffs
{% endhighlight %}

Now a function is made which defines the extent of each tile and also checks which input rasters are needed to create it. A critical part of this is identifying if a tile intersects more than one raster. Depending on your particular situation this could be every tile or none of them. It is important to identify these tiles as they will need to be dealt with slightly differently than the other tiles.

{% highlight python %}
# will take tile bounds and only export them if they fall within a geotiff
# this is called row by row by pool below
def make_polygons(row):
    tile_polygon_list = []
    tile_top = bound_y_max + y_move*row
    tile_bottom = tile_top + y_tile_size
    tile_left = bound_x_min

    for col in range(0,number_of_cols):
        tile_left = bound_x_min + col*x_move
        tile_right = tile_left + x_tile_size
        tile_dict = {'top':tile_top,'left':tile_left,'bottom':tile_bottom,'right':tile_right}
        tile_list = np.array([tile_top,tile_left,tile_bottom,tile_right])
#         check if valid tile
        intersect = intersect_tile_with_geotiffs(tile_dict,geo_tiff_bounds)
        raster_name = raster_name_append+str(row)+'_'+str(col)+'.tif'
        if len(intersect) > 0:
            polygon = {'geometry':geometry.Polygon([[tile_left, tile_top], [tile_right, tile_top], [tile_right, tile_bottom], [tile_left, tile_bottom]]),
                      'intersect':intersect, 'row':row, 'col':col, 'name':raster_name}
            tile_polygon_list.append(polygon)
    return tile_polygon_list
{% endhighlight %}

Yet again, this cell uses multiprocessing to call multiple copies of the above function at the same time.

{% highlight python %}
# multiprocess making polygons
with Pool() as pool:
    tile_polygon_list = list(tqdm(pool.imap(make_polygons, range(0,number_of_rows)), total=len(range(0,number_of_rows))))

# this is returned as a list of list so it must be flattened
tile_polygon_list = list(np.concatenate(tile_polygon_list).ravel())
{% endhighlight %}

Now that we have the extent of each tile, a geodataframe is created. This geodataframe is used to visualise the extent of each tile which can be useful if something unexpected is happening. This geodataframe contains the bounds of each tile as well as its row, column and name. In the next couple of cells, this geodataframe is displayed in multiple different ways and then exported out to disk.

{% highlight python %}
#  convert into geodataframe
polygon_tiles_gpd = gpd.GeoDataFrame(tile_polygon_list,geometry='geometry',crs=crs)
del polygon_tiles_gpd['intersect']
polygon_tiles_gpd
{% endhighlight %}

{% highlight python %}
# plot up a bunch of polygons
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
polygon_tiles_gpd.head(10000).plot(column='col', ax=ax, legend=True,cax=cax);
{% endhighlight %}

{% highlight python %}
# plot a small number of polygon bounds
polygon_tiles_gpd.head(10).boundary.plot()
{% endhighlight %}

{% highlight python %}
polygon_export_loc = os.path.join(os.path.dirname(output_folder),'output_no_overlap.gpkg')
polygon_export_loc
{% endhighlight %}

{% highlight python %}
polygon_tiles_gpd.to_file(polygon_export_loc, driver="GPKG")    
{% endhighlight %}

A naive approach to cutting tiles may be to open a tile, open each intersecting input raster, then cut out the necessary parts of each raster, join the parts together and move to the next tile. This would work but would result in each input raster being read multiple times which could be quite slow. Instead, if each input raster is opened and multiple tiles are cut one after another, each raster would only need to be opened once, reducing load times. To operate in this manner, the below function is necessary. The function identifies which tiles intersect each input raster. Fortunately, most of this work is already done in the intersecting we performed above, so this function just looks through the geodataframe to work out the intersects.

{% highlight python %}
# make a list of which tiles are within which geotiffs
def intersector(geo_tiff):
    tiles_inside_geo_tiff = []
#     loop over each tile and check if the geotiff is the the intersect list
    for tile in tile_polygon_list:
        if geo_tiff in tile['intersect']:
#             count this so we know if the tile will be incomplete or not
            incomplete = len(tile['intersect'])>1
#             build dict with geom the current row and col for naming
            tiles_inside_geo_tiff.append({'geometry':tile['geometry'],'row':tile['row'],'col':tile['col'],'name':tile['name'],'incomplete':incomplete})
    return([geo_tiff,tiles_inside_geo_tiff])
{% endhighlight %}

This intersect function is called using multiprocessing

{% highlight python %}
with Pool() as pool:
    geo_tiff_with_tiles = list(tqdm(pool.imap(intersector, geo_tiff_list), total=len(geo_tiff_list)))
{% endhighlight %}

The tile cutting function is now defined, this function takes a list as an input, the first element being a path to a raster and the second being a list of tiles to cut. The raster is opened and then each tile is cut out as long as it does not already exist. This function also checks that the tile being cut does not also intersect another raster, if so the tiles filename is marked with the string ‘_incomplete.tif’. At the end of the function, a list of incomplete tiles is returned.

{% highlight python %}
# cut tiles from rasters
def cut_tiles(geotiff):
#     grab path to to file and open it
    geotiff_open = gdal.Open(geotiff[0])
#     grab the filename and strip the extension
    geo_tiff_filename = os.path.basename(geotiff[0]).replace(input_file_ext,'')
    incomplete_tile_list = []
    for tile in geotiff[1]:
        tile_geometry = tile['geometry']
#         shapely bounds returns "minx, miny, maxx, maxy" but we need minx, maxy, maxx, miny
        top = list(tile_geometry.bounds)[3]
        bottom = list(tile_geometry.bounds)[1]
        left = list(tile_geometry.bounds)[0]
        right =list(tile_geometry.bounds)[2]

#         make row folder path
        if output_to_row_folders:
            output_row_folder = os.path.join(output_folder,str(tile['row']))
            Path(output_row_folder).mkdir(parents=True, exist_ok=True)
        else:
            output_row_folder = output_folder
#       make row folder if necessary

        export_file_name = str(tile['name'])#str(tile['row'])+'_'+str(tile['col'])+'.tif'

#         check if tile is incomplete if so append the getiff name so that it is unique
        if tile['incomplete']:
            append_name = '-'+geo_tiff_filename+'_incomplete.tif'
            export_file_name = export_file_name.replace('.tif',append_name)
#             add tile to list so we don't need to re-find them to compile incomplete tiles
            export_file_path = os.path.join(output_row_folder,export_file_name)
            incomplete_tile_list.append(export_file_path)
        else:
            export_file_path = os.path.join(output_row_folder,export_file_name)

#         check if already done
        if not os.path.isfile(export_file_path):

    #     clip the data
    #         make a string of tile dims to pass as a command line arg, this is kinda of hacky, would like a better option
            tile_clip_string = str(left) +' '+str(top) +' '+str(right) +' '+str(bottom)

            translate_options = gdal.TranslateOptions(gdal.ParseCommandLine("-projwin "+tile_clip_string)
                                                     ,creationOptions=['COMPRESS='+output_compression])

            tile_clip = gdal.Translate(export_file_path, geotiff_open, options = translate_options)
    #     close the tile
            tile_clip = None
    return incomplete_tile_list
{% endhighlight %}

Now another function is defined which will split a list into roughly equally sized sublists this is necessary to batch the tiling process.

{% highlight python %}
# used to slip a list into n chunks of roughly the same size
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
{% endhighlight %}

This cell simply counts the number of CPU cores available.

{% highlight python %}
concurrent_threads = cpu_count()
concurrent_threads
{% endhighlight %}

This cell works out the average number of tiles each input raster will be cut into.

{% highlight python %}
tile_count = 0
for raster,tiles in geo_tiff_with_tiles:
    tile_count+=len(tiles)
average_tile_count = round(tile_count/len(geo_tiff_with_tiles))
average_tile_count
{% endhighlight %}

Depending on your CPU core count and the number of input rasters there are two different approaches, both optimised for speed in different situations. For example, if you only have one input raster and 10 CPU cores then each core will get a copy of the same input raster and a sublist of tiles to cut out. However, if you have 10 CPU cores and 100 input rasters then each core will receive its own raster, cutting out all of that rasters tiles, then moving on to the next raster. Again all of this is done using multiprocessing for speed.

{% highlight python %}
# if you have more cores than rasters and more cores than average tile count then give every core a copy of the same raster to work on
if concurrent_threads > len(geo_tiff_with_tiles) and concurrent_threads < average_tile_count:
    print('You have more threads than rasters so we each thread will get the same raster')
    incomplete_tile_list = []

    for raster,tiles in tqdm(geo_tiff_with_tiles):
        geo_tiff_with_tile_list = []

        for tile_split in split(tiles,concurrent_threads):
            geo_tiff_with_tile_list.append([raster,tile_split])

        pool = Pool(concurrent_threads)
        with pool:
            one_raster_incomplete_tile_list = list(pool.imap(cut_tiles,geo_tiff_with_tile_list))

        incomplete_tile_list.append(np.concatenate(one_raster_incomplete_tile_list))

else:
    print('you have more rasters than threads so each thread will get its own raster')
    pool = Pool()
    with pool:
        incomplete_tile_list = list(tqdm(pool.imap(cut_tiles,geo_tiff_with_tiles), total=len(geo_tiff_with_tiles)))
{% endhighlight %}

The returned ‘incomplete_tile_list’ above is a list of lists, this cell is used to flatten it into one long list.

{% highlight python %}
flat_incomplete_tile_list = np.concatenate(incomplete_tile_list).ravel()
print(len(flat_incomplete_tile_list),'incomplete tiles')
{% endhighlight %}

The next two cells build a dataframe that contains two columns, one containing the paths to the incomplete tiles and the other containing the row and column for that tile.

{% highlight python %}
incomplete_tile_file_names = []
for incomplete_tile in flat_incomplete_tile_list:
    incomplete_tile_file_names.append(os.path.basename(incomplete_tile).split('-')[0].replace(raster_name_append,''))
{% endhighlight %}

{% highlight python %}
incomplete_tile_df = pd.DataFrame(
    {'incomplete_tiles': flat_incomplete_tile_list,
     'row_col': incomplete_tile_file_names
    })

unique_tiles_list = incomplete_tile_df.row_col.unique()

incomplete_tile_df.head()
{% endhighlight %}

This cell is quite handy and I actually use it quite a bit outside of this script as well. It utilises the ‘gdal_merge.py’ python script to drastically simplify the GDAL merging process.

{% highlight python %}
def merge_tiles(merge_imgs, output_path):
    merge_command = ['gdal_merge.py', '-o', output_path, '-quiet', '-co','COMPRESS='+output_compression]
    for name in merge_imgs:
        merge_command.append(name)
    subprocess.run(merge_command)
{% endhighlight %}

This cell takes a row and column and finds all the incomplete tiles associated with that location,  which are then sent to the merging script. When the merging is completed, the incomplete files are removed as they are no longer required.

{% highlight python %}
def join_incomplete_tile(unique_tiles):
#     filter the dataframe down to only one row and col
    df_filt = incomplete_tile_df[incomplete_tile_df['row_col']==unique_tiles]
#     get paths as list
    combine_these = df_filt['incomplete_tiles'].tolist()
#     get export name by removing the geotiff names from end
    export_file_name = os.path.basename(combine_these[0]).split('-')[0]+'.tif'
#     grab the folder from the first tile
    export_dir_path = os.path.dirname(combine_these[0])
#     build full export path
    export_full_path = os.path.join(export_dir_path,export_file_name)
#     use gdal gdal_merge.py to merge the tiles
    merge_tiles(combine_these,export_full_path)
#     remove the incomplete tiles and msk files
    for incomplete_tile in combine_these:
        try:
            os.remove(incomplete_tile)
        except:
            print('could not remove')
        if os.path.isfile(incomplete_tile+'.msk'):
            try:
                os.remove(incomplete_tile+'.msk')
            except:
                print('could not remove')   
{% endhighlight %}

Finally, multiprocessing is called one last time to join incomplete tiles.

{% highlight python %}
pool = Pool()
with pool:
    list(tqdm(pool.imap(join_incomplete_tile,unique_tiles_list), total=len(unique_tiles_list)))
{% endhighlight %}

Now the notebook is completed, if you navigate to your output folder you will find a collection of subfolders. Each subfolder contains one row of tiles. It is broken up like this as many file navigators will start to slow down if an open folder contains many thousands of images.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/How%20to%20efficiently%20create%20millions%20of%20overlapping%20raster%20tiles.ipynb">link to notebook</a>
