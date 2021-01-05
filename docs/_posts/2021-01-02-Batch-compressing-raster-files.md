---
layout: post
title: Batch compressing raster files
---
When I’m finished with a project, I often have a large amount of raster files on disk which I need to archive for later use. Depending on the structure of the data, it can be highly advantageous to compress the data before storing it away. I have put together some Python code in a Jupyter Notebook to compress all raster files in a folder to streamline the task.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Batch%20compressing%20raster%20files.ipynb">link to notebook</a>

The script starts by importing a bunch of modules, “os” for manipulating file paths, “glob” for grabbing a list of files in a folder, “Pool” for multiprocessing, “tqdm” for displaying a loading bar, “Path” for making folders and lastly “gdal” for raster manipulation.

{% highlight python %}
import os
import glob
from multiprocessing import Pool
from tqdm.auto import tqdm
from pathlib import Path
from osgeo import gdal
{% endhighlight %}

You need to specify the folder which contains the uncompressed rasters, and the folder which will contain the compressed rasters. The script will then use glob to search the “tif_dir” for rasters ending with “.tif” and place the paths to these files in a list called list_tifs.

{% highlight python %}
tif_dir = '/home/nick/Desktop/full'

export_dir = '/home/nick/Desktop/full comp'
Path(export_dir).mkdir(parents=True, exist_ok=True)

list_tifs = glob.glob(tif_dir+"/*.tif")
print('Image count', len(list_tifs))
{% endhighlight %}

The compression options are selected with “gdal.TranslateOptions()”. For most uses “LZW” with “PREDICTOR=2” is a good compromise of speed and compression ratio, however you may wish to change these depending on your specific situation.
For some context check out these links:

<a href="https://digital-geography.com/geotiff-compression-comparison/">Geotiff compression comparison</a>

<a href="https://kokoalberti.com/articles/geotiff-compression-optimization-guide/">Geotiff compression optimization guide</a>

{% highlight python %}
topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2', 'NUM_THREADS=ALL_CPUS','GDAL_DISABLE_READDIR_ON_OPEN=TRUE'])
{% endhighlight %}

The next cell sets up a function to do the heavy lifting. Firstly, it extracts the file name of the input tif path with “os.path.basename()” then constructs the “export_path” using the supplied “export_dir” and tif basename. Then it will check to see if the export file already exists, this is useful if you have already run this script but it did not complete. The script will only make the file if it does not already exist.

Now the function pulls in the compression options with a gdal.Translate() which opens the uncompressed tif and re-saves it with the new compression in the specified folder.

{% highlight python %}
def compress_tifs(tif):
    file_name = os.path.basename(tif)
    export_path = os.path.join(export_dir,file_name)

    if not os.path.isfile(export_path):
        gdal.Translate(export_path, tif, options=topts)
{% endhighlight %}

Finally we call “tqdm(Pool().imap()” which runs the compression function using multiprocessing and displays a handy loading bar so you can see your progress.

{% highlight python %}
with Pool() as p:
    list(tqdm(p.imap(compress_tifs, list_tifs), total=len(list_tifs)))
{% endhighlight %}

Keep in mind that depending on your data, storage medium and CPU power you may be better off limiting the amount of simultaneous processes. You can experiment with this by placing the number of desired processes within the brackets of the “Pool()” call. This can be particularly useful if you are reading from, or exporting to, a hard drive as having too many processes may slow you down by making the drive spend more time moving around to find the location of your data and less time actually reading it.

If you are wanting to compress RGB rasters and you are fine with lossy compression you may also want to check out “JPEG” compression (yes you can have .jpg compression within a .tif). This will massively reduce the image size with a slight loss of detail which may be worthwhile in some cases.

<a class="jn" href="https://github.com/DPIRD-DMA/blog/blob/master/notebooks/Batch%20compressing%20raster%20files.ipynb">link to notebook</a>
