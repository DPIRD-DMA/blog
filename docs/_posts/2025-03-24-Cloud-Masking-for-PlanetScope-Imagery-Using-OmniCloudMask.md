---
layout: post
title: "Cloud Masking for PlanetScope Imagery Using OmniCloudMask"
---

As a remote sensing analyst, you know the frustration of cloud contamination in satellite imagery. While Planet's UDM2 masks provide basic cloud detection, they often miss thin clouds and shadows that can compromise your analysis. We developed OmniCloudMask (OCM) as a solution that achieves significantly higher accuracy than UDM2, with balanced overall accuracy values of 96.9% for clear areas, 98.8% for clouds, and 97.4% for shadows on PlanetScope data.

The best part? OCM wasn't even trained on PlanetScope data! It generalises effectively from its Sentinel-2 training data to work across multiple sensors, giving you professional-quality cloud masks for your PlanetScope analysis pipeline.

OCM is optimised for both accuracy and throughput, processing the average PlanetScope scene in just one to two seconds with a modern NVIDIA GPU.

OCM works with various input types as long as you have the red, green, and NIR bands, and the spatial resolution is between 10 m and 50 m (for higher resolution data, simply resample the input to 10 m). OCM also works regardless of processing level, making it highly adaptable.

Recently [NASA has been using OCM](https://github.com/nasa-nccs-hpda/vhr-cloudmask.pytorch) for processing Maxar imagery 🚀

[The Paper 📜](https://www.sciencedirect.com/science/article/pii/S0034425725000987)

[The GitHub project 💻](https://github.com/DPIRD-DMA/OmniCloudMask)

<div class="image-container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
    <div class="image-panel" style="margin-right: 10px; width: 32%;">
        <img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2025-03-24-Cloud-Masking-for-PlanetScope-Imagery-Using-OmniCloudMask_1.png?raw=true" alt="PlanetScope RGB Image" style="width: 100%;">
        <p style="text-align: center; font-style: italic;">Original RGB</p>
    </div>
    <div class="image-panel" style="margin: 0 10px; width: 32%;">
        <img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2025-03-24-Cloud-Masking-for-PlanetScope-Imagery-Using-OmniCloudMask_2.png?raw=true" alt="PlanetScope with UDM2 Mask" style="width: 100%;">
        <p style="text-align: center; font-style: italic;">RGB + UDM2</p>
    </div>
    <div class="image-panel" style="margin-left: 10px; width: 32%;">
        <img src="https://github.com/DPIRD-DMA/blog/blob/master/docs/images/blog_images/2025-03-24-Cloud-Masking-for-PlanetScope-Imagery-Using-OmniCloudMask_3.png?raw=true" alt="PlanetScope with OCM Mask" style="width: 100%;">
        <p style="text-align: center; font-style: italic;">RGB + OCM</p>
    </div>
</div>


## Getting Started with OCM for PlanetScope

Installation is straightforward:

```bash
pip install omnicloudmask
```

Once installed, you can start processing your PlanetScope imagery immediately.

## Basic Usage for Individual Images

Here's a simple approach to mask a single PlanetScope image:

```python
from omnicloudmask import predict_from_array,load_multiband
from functools import partial
from pathlib import Path
import rasterio as rio

planetscope_image = Path("path/to/planetscope_image.tif")

# Make sure you collect the correct bands (red, green, NIR)
# https://developers.planet.com/docs/apis/data/sensors/

# If you have an older 4 band image use this band order
band_order = [3, 2, 4]

# If you have a newer 8 band image use this band order
# band_order = [6, 4, 8]

# Open and resample input to 10 m
rgn_data, profile = load_multiband(input_path=planetscope_image, resample_res=10, band_order=band_order)

# Generate cloud and shadow mask
prediction = predict_from_array(rgn_data)

# Save the result
profile.update(count=1, dtype='uint8')
with rio.open("planetscope_cloud_mask.tif", 'w', **profile) as dst:
    dst.write(prediction.astype('uint8'))
```

The resulting mask will have these class values:

- 0 = Clear
- 1 = Thick Cloud
- 2 = Thin Cloud
- 3 = Cloud Shadow

## Processing Multiple PlanetScope Images

For batch processing

```python
from pathlib import Path
from omnicloudmask import predict_from_load_func, load_multiband
from functools import partial

# List of PlanetScope image paths
scene_paths = list(Path("planet_data_directory").glob("*.tif"))

# Make sure you collect the correct bands (red, green, NIR)
# https://developers.planet.com/docs/apis/data/sensors/

# If you have an older 4 band image use this band order
band_order = [3, 2, 4]

# If you have a newer 8 band image use this band order
# band_order = [6, 4, 8]

load_planetscope = partial(load_multiband,resample_res=10, band_order=[3,2,4])
# Process all scenes
prediction_paths = predict_from_load_func(
    scene_paths,
    load_planetscope,
    batch_size=4
    )

print(f"Cloud masks saved to: {prediction_paths}")
```

## Advanced Configuration for GPU Acceleration

If you're processing large volumes of imagery, OCM offers GPU acceleration options:

```python
# For NVIDIA GPUs
prediction_paths = predict_from_load_func(
    scene_paths,
    load_planetscope,
    batch_size=8,               # Higher for faster processing
    inference_device="cuda",    # Use NVIDIA GPU
    inference_dtype="bf16",     # Use BF16 precision if supported
)

# For systems with limited VRAM
prediction_paths = predict_from_load_func(
    scene_paths,
    load_planetscope,
    inference_device="cuda",    # Use GPU for model inference
    inference_dtype="bf16",     # Use BF16 precision if supported
    mosaic_device="cpu"         # Use CPU for patch merging (less VRAM intensive)
)
```

## Conclusion

With OCM, you now have access to state-of-the-art cloud and shadow masking for PlanetScope imagery. This dramatically improves the reliability of your automated analyses and reduces the manual effort needed to filter out cloud-contaminated data.

For more information and advanced usage examples, check out the OmniCloudMask GitHub repository or [read the paper](https://www.sciencedirect.com/science/article/pii/S0034425725000987)
 detailing how OCM was developed.

Happy cloud-free analysis! 🛰️☁️