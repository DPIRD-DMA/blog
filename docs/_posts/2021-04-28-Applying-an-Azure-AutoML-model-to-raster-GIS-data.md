---
layout: post
title: Applying an Azure AutoML model to raster GIS data
---


This is a walk through of a Jupyter Notebook I created to run a vegetation classification model over the Nullarbor Land Conservation District. This notebook assumes you are trying to execute your own geographic classification task and that you already have a trained model from Azure AutoML. It is also assumed that your input raster data is already prepared and all your data has the same extent, pixel size and projection. For my particular application I was using 100 raster layers at 80m resolution, which covers 150,000 km2 and equates to about 20,000,000 pixels in total.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Point%20sampling%20multiple%20raster%20files.ipynb">link to notebook</a>

The onerous but necessary setup:
Warning, these setup steps are likely to change over time as Azure updates their dependencies. These particular instructions were valid as of 28 April 2021.

If you haven't already, go ahead and download your model from Azure. You should receive a zip archive with three files, extract them into a new folder. Within your new folder you should have “conda_env_v_1_0_0.yml” the conda environment setup file, “model.pkl” the trained model and “scoring_file_v_1_0_0.py” which contains an example of how to use the model.

Before moving ahead, it is necessary to set up a conda environment with the specific libraries necessary to run the Azure model. The ‘conda_env_v_1_0_0.yml’ file you have just extracted contains most of what you need to make the environment, however you do need to make some slight tweaks. Open the ‘conda_env_v_1_0_0.yml’ file with a text editor and add ‘- pip’ above python, then set the python version number to 3.7.

Next, open a terminal and ‘cd’ into your extracted folder, then run the following command to build your environment:

conda env create -f conda_env_v_1_0_0.yml

Your environment now contains all of the Azure specific ML libraries, however you still need to update some of packages and add a couple extra libraries. Run the following commands one by one to finish off the install:

conda activate project_environment
conda install -c conda-forge rasterio
conda install -c conda-forge tqdm
conda install -c conda-forge jupyter
conda update --all

Your environment is now set up. The following commands will change your current directory back to the root directory and then open Jupyter.

cd /
jupyter notebook

The ‘fun’ part:
After running the last command above, you should see a web browser window open. Within this window, navigate to the Jupyter Notebook that you have downloaded from here {link}.

Try running the first cell which imports all the necessary libraries. If this executes correctly your environment is probably set up correctly and you can move on to the second cell.

{% highlight python %}
# system  
import os
from tqdm.auto import tqdm
import pickle
import joblib

# data
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# Azure
import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
{% endhighlight %}
The second cell requires you to input the paths to your raster dataset, the raster file format, your desired location of the output raster file and the path to the Azure model file.

{% highlight python %}
# point this at your folder of raster covariates
raster_folder  = '/media/nick/2TB Working 2/Projects/Nulla ML test/Cov_nulla - for Nick'
# what raster file format are your covariates?
input_file_ext = '.tif'
# this is your predicted raster layer
output_raster_file = '/media/nick/2TB Working 2/Projects/Nulla ML test/Nulla_nick_13-4-2021_DL.tif'
# this is the path to the 'model.pkl' file downloaded from Azure
model_path = '/media/nick/2TB Working 2/Projects/Nulla ML test/DL model v5 DL/AutoML880de06d80/model.pkl'
{% endhighlight %}

The third cell performs some sanity checks on the above data and lets you know if it sees any issues.


{% highlight python %}
print("Raster folder valid?",os.path.isdir(raster_folder))
print("Model file valid?",os.path.isfile(model_path))
if os.path.isfile(output_raster_file):
    print('WARNING the output file already exists....')
{% endhighlight %}

It is possible that your raster data has some value/values which represents NaN/null/no data, which needs to be addressed as your trained model will not enjoy dealing with those values. If you built your raster dataset, you probably know what this value is but if not, load some of your rasters into QGIS or alike and find out. It’s probably something like -9999 or -3.3e+38. If you have one or more of these values, enter them as a list, if you don’t have any of them just leave ‘raster_nan_values’ as an empty list.
There are two options for dealing with these NaN values. You can mask them out of the final product and/or you can replace the NaN values with the layer medians. Either or both of these approaches can be used. If you set the ‘nan_threshold’ to a number higher than 0, each pixel must have that many NaN values before it is masked out. This is useful if you have inconsistent raster coverage. However, keep in mind the ability of the model to make correct predictions may be significantly degraded in areas which have had NaN values replaced with medians.
Finally, set the data type of the output raster, ‘uint8’ should almost always be sufficient for this purpose.

{% highlight python %}
# if your rasters have a NaN value (or multible) set it here
raster_nan_values = [-3.3999999521443642e+38]
# the max number of nan values at one location before that location is masked out
nan_threshold = 1
# what data type do you want the export file to be? (this must be compatible with numpy and rasterio/gdal)
# uint8 should be fine upto 256 classes
data_dtype = 'uint8'
{% endhighlight %}

In this next cell you need to enter the classification classes that you trained on. They do not need to be in any particular order, just keep in mind the output raster will have these labels replaced with numbers.
You must also specify an output nan value as a ‘uint8’ valid number. The default is set to 255, which should be fine for most cases.

{% highlight python %}
# what classes do you have? the raster will be labeled in this order
classes = ['Bindii','Eucch','Nulbr3','Nulbr1','Nulbr2','Spwgr']
# this must be an int value compatible with your 'data_dtype'
export_raster_nan = 255
print("Your export raster NaN value is",export_raster_nan)
{% endhighlight %}

This next cell searches the defined ‘raster_folder’ and subfolders and returns a list of files which end with the ‘input_file_ext’. This is useful as your raster folder structure can be nested or flat and this will work regardless.

{% highlight python %}
# search folder and all sub folders for 'input_file_ext' files
raster_list = []
for root, dirs, files in os.walk(raster_folder):
    for file in files:
        if file.endswith(input_file_ext):
            raster_list.append(os.path.join(root, file))

print('Number of rasters found',len(raster_list))
{% endhighlight %}

The next step is to add the raster data into a pandas dataframe. Depending on the size of your raster data, this step may soak up all of your RAM so keep an eye on your RAM usage in System Monitor/Task Manager/Activity Monitor (depending on your OS). If you find yourself running out of RAM, your only choice is to tile your input rasters before running this script.
This cell starts by creating an empty Pandas dataframe and then iterates over each found raster and extracts the first band of data. At this point the defined no data values within each raster are replaced with ‘numpy.nan’ values before being added to the dataframe.

{% highlight python %}
# loop over each raster, extract the data and place in dataframe
# this is rather RAM hungry, you may need to pre tile the data if it will not fit
# make blank dataframe
joined_data = pd.DataFrame()
# this could be multithreaded but it's normally pretty quick as is
# loop over rasters
for raster_path in tqdm(raster_list):
#     grab the raster name to use as column heading
    raster_file_name = os.path.basename(raster_path).replace(input_file_ext,'')
#     open raster with rasterio and read first band
    geotiff_open = rasterio.open(raster_path).read(1)
#     convert data into 1D numpy array and convert to float32 to avoid Skilearn errors
    raster_as_1D_array = np.array(geotiff_open).flatten().astype('float32')
#     replace raster nan values with numpy nan values
    if len(raster_nan_values)>0:
        for raster_nan_val in raster_nan_values:
            raster_as_1D_array[raster_as_1D_array == raster_nan_val] = np.nan
#     place data into dataframe
    joined_data[raster_file_name] = raster_as_1D_array
{% endhighlight %}

Now that all the raster data is in the dataframe, the smallest and largest values are printed to make sure no NaN values were missed. Assuming these numbers look reasonable, move on to the next cell.

{% highlight python %}
# print out the smallest and largest numbers in the dataframe, make sure you dont have any crazy numbers here
# this can be a little slow but can be handy to debug bad data
print("Smallest number is", min(list(joined_data.min())))
print("Largest number is", max(list(joined_data.max())))
{% endhighlight %}

In this cell the number of NaN values in each row in the dataframe is counted.  If this number is above the NaN threshold it gets a True tag in the ‘Mask’ column and will be masked out later.

{% highlight python %}
# count the frequency of NaN values in each row
joined_data['nan_count'] = joined_data.isnull().sum(axis=1)
# apply threshold to the 'nan_count', if a row has more NaNs than the 'nan_threshold' it will be masked in the output
joined_data['mask'] = joined_data['nan_count']>nan_threshold
Joined_data
{% endhighlight %}

It is now time to actually replace the NaN values with something that the model will accept. The solution here is to simply replace every NaN value with the median for that column/raster layer. You could also use mean, however this could create problems if you have a column of integers or booleans.
This step should be possible with a one-liner, however I found this to use a lot of RAM so instead a for loop is used.


{% highlight python %}
# replace all nan values with column median
# you should be able to do this in one step but it uses too much RAM, so a loop instead
# joined_data.fillna(joined_data.median(), inplace=True)
# loop over each column and apply median to NaNs
for column in tqdm(joined_data,total = joined_data.shape[1]):
    joined_data[column] = joined_data[column].fillna(joined_data[column].median())
joined_data
{% endhighlight %}

Now that the mask column is completed, this cell extracts the mask as a list and then removes the ‘mask’ and the ‘nan_count’ columns from the dataframe. This is done  to prevent the model from returning an error if it passes extra columns that it was not trained on.


{% highlight python %}
# get a list of the mask pixels
mask_values_list = joined_data['mask'].to_list()
# remove mask and NaN cols from dataframe, as we don't want to pass them into the model
del joined_data['mask']
del joined_data['nan_count']
{% endhighlight %}

This next cell requires you to open up  ‘scoring_file_v_1_0_0.py’ file which should be in your extracted folder from Azure. You should see a dataframe called ‘input_sample’ just under the import lines, which contains all your training data variable names. Copy this entire block of code and past it into the current Jupyter Notebook cell, overwriting the current ‘input_sample’ line.

{% highlight python %}
# this is from the Azure 'scoring_file_v_1_0_0.py' file, just copy your version of this line in here
# it will help us make sure you have all the necessary raster data
input_sample = pd.DataFrame({"Clim_etaaann": pd.Series([0], dtype="int64"), "Clim_etaajan": pd.Series([0.0],.......pd.Series([0.0], dtype="float64")})
{% endhighlight %}

The next two cells check that the raster dataframe and the ‘input_sample’ have the same column headings. Again, this is critical, otherwise the model will throw an error.


{% highlight python %}
# columns in training data which are not in your raster data
for i in input_sample.columns:
    if i not in joined_data.columns:
        print(i)
{% endhighlight %}

{% highlight python %}
# columns in your raster data which were not in the training data
for i in joined_data.columns:
    if i not in input_sample.columns:
        print(i)
{% endhighlight %}

At this point the raster dataframe could be passed into the pickled model, however considering the size of the input data it may take a long time for the model to complete. This cell breaks the raster dataframe up into chunks of ‘n’ length, which will be processed one by one so that you can see some sense of progress.

{% highlight python %}
# break the dataframe into chunks of n length so we can track progress
n = 50000
list_df = [joined_data[i:i+n] for i in range(0,joined_data.shape[0],n)]
{% endhighlight %}

Everything is now ready for the model to be run. As the dataframe is now in chunks, the model is called in a for loop which will be tracked with ‘tqdm’. The output is a list of lists with each item being one of your defined classes.


{% highlight python %}
# run the model on each chunk and append the result to a list
# dont stress about the "Failure while loading azureml_run_type_providers" warning
preds_list = []
# load model from pickel on disk
model = joblib.load(model_path)
for df_chunk in tqdm(list_df):
    preds_list.append(model.predict(df_chunk))
{% endhighlight %}

As the final export file is going to be a raster, the predictions must be converted to integers. This is done by looping over each item in each sublist and finding the index of that item in the ‘classes’ list. The result of this is appended to ‘pred_nums’.

{% highlight python %}
# the loop above builds a list of lists, the loop below will flatten that into one long list
# the model outputs class names, but to make a raster we need ints instead so use 'index' to find the corresponding int
pred_nums = []
for chunk in tqdm(preds_list):
    for i in chunk:
        pred_nums.append(classes.index(i))
{% endhighlight %}

It is now time to use the mask list we created previously, which  is combined into a new dataframe with the modeled predictions. This cell also replaces any predictions which need to be masked by the predefined ‘export_raster_nan’ value.

{% highlight python %}
# build a new dataframe with the prediction ints and the mask list
preds_mask_dict = {'pred_nums':pred_nums,'mask':mask_values_list}
pred_df = pd.DataFrame(preds_mask_dict)
# wherever we have a mask True value replace the pred_nums with the 'export_raster_nan' value
pred_df.loc[pred_df['mask'],'pred_nums'] = export_raster_nan
pred_df
{% endhighlight %}
To export the predictions into a geotiff, the geographic metadata must first be defined. This could be done manually, however, this cell instead opens one of the input rasters and extracts most of the info that is required.

{% highlight python %}
# open one raster so we can copy its metadata for the export
src = rasterio.open(raster_list[0])
# grab a copy of the metadata we will use this to export the predictions
raster_meta = src.meta
# grab a copy of the input raster shape
raster_shape = src.shape
{% endhighlight %}

This next cell makes a couple of changes to the metadata to reduce file size and to set the correct no data value.

{% highlight python %}
# set the dtype to reduce file size
raster_meta['dtype'] = data_dtype
# set the no data value
raster_meta['nodata'] = export_raster_nan
# set LZW compression to reduce file size
raster_meta['compress'] = 'LZW'
raster_meta
{% highlight python %}

Now that the metadata has been extracted, the prediction list can be reshaped into a 2D array to match the input rasters.



{% highlight python %}
# get dataseries of the predicted numbers including the NaN mask
pred_nums_masked = pred_df['pred_nums']
# reshape the predictions list to the same as input raster
preds_reshaped = np.array(pred_nums_masked).reshape(raster_shape)
{% highlight python %}

Now that the array has been reshaped, it can be visualized with ‘matplotlib’. This is a little more complicated than normal because we have a no data value, so ‘np.ma.masked_where’ is used to manually set the ‘export_raster_nan’ value to white.

{% highlight python %}
# visualize the 2D array
cmap = plt.cm.viridis
cmap.set_bad(color='white')
plt.imshow(np.ma.masked_where(preds_reshaped == export_raster_nan, preds_reshaped), interpolation='nearest',cmap=cmap)
plt.show()
{% highlight python %}

The final cell simply exports out the 2D array to the location defined by ‘output_raster_file’.

{% highlight python %}
# write out the 2D array file to disk
with rasterio.Env():
    with rasterio.open(output_raster_file, 'w', **raster_meta) as dst:
        dst.write(preds_reshaped.astype(data_dtype), 1)
{% highlight python %}

You should now be able to navigate to the export location and drag the geotiff into QGIS to visualise it, hopefully it looks reasonable!

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Point%20sampling%20multiple%20raster%20files.ipynb">link to notebook</a>
