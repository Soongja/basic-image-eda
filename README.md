# basic-image-eda

A simple multiprocessing EDA tool to check basic information of images under a directory(images are found recursively). This tool was made to quickly check info and prevent mistakes on reading, resizing, and normalizing images as inputs for neural networks. It can be used when first joining an image competition or training CNNs with images!

*Notes:*  
\- All images are converted to 3-channel(rgb) images. When images that have various channels are mixed, some results can be misleading.  
\- uint8 and uint16 data types are supported. If different data types are mixed, error occurs.  
\- Supported extensions: jpg, jpeg, jpe, png, tif, tiff, bmp, ppm, pbm, pgm, sr, ras, webp  

### Installation
```bash
pip install basic-image-eda
```
or (latest version)
```bash
pip install git+https://github.com/Soongja/basic-image-eda
```

prerequisites:
- opencv-python
- numpy
- matplotlib
- skimage.io
- tifffile
- tqdm

### Usage(CLI/Code)
#### CLI
simple one line command!
```bash
basic-image-eda <data_dir>
```
or
```bash
basic-image-eda <data_dir> -e png tiff -t 12 --dimension_plot --channel_hist --nonzero --hw_division_factor 2.0 > eda.txt

Options:
  -e --extensions          target image extensions. if none, all supported extensions are included.(default=None)
  -t --threads             number of multiprocessing threads. if 0, automatically count max threads.(default=0)
  -d --dimension_plot      show dimension(height/width) scatter plot.(default=False)
  -c --channel_hist        show channelwise pixel value histogram. takes longer time.(default=False)
  -n --nonzero             calculate values only from non-zero pixels of the images.(default=False)
  -f --hw_division_factor  divide height,width of the images by this factor to make pixel value calculation faster.
                           Information on height, width are not changed and will be printed correctly.(default=1.0)
  -V --version             show version.
```

#### Code
```python
from basic_image_eda import BasicImageEDA

if __name__ == "__main__":  # for multiprocessing
    data_dir = "./data"
    BasicImageEDA.explore(data_dir)
        
    # or
    
    extensions = ['png', 'jpg', 'jpeg']
    threads = 0
    dimension_plot = True
    channel_hist = True
    nonzero = False
    hw_division_factor = 1.0
    
    BasicImageEDA.explore(data_dir, extensions, threads, dimension_plot, channel_hist, nonzero, hw_division_factor)
```

### Results
#### Results on [celeba dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (test set)

<table border="0">
<tr>
    <td>
    <img src="https://user-images.githubusercontent.com/32871371/81141998-43ebe700-8fa9-11ea-9645-fff2cc83ab9b.png" width="100%">
    </td>
    <td>
    <img src="https://user-images.githubusercontent.com/32871371/81142025-5fef8880-8fa9-11ea-98eb-2c43b256fa8d.png", width="100%">
    </td>
</tr>
</table>

```
found 19962 images.
Using 12 threads. (max:12)

*--------------------------------------------------------------------------------------*
number of images                         |  19962

dtype                                    |  uint8
channels                                 |  [3]
extensions                               |  ['jpg']

min height                               |  85
max height                               |  5616
mean height                              |  591.8215108706543
median height                            |  500

min width                                |  85
max width                                |  5616
mean width                               |  490.2976655645727
median width                             |  396

mean height/width ratio                  |  1.207065732587525
median height/width ratio                |  1.2626262626262625
recommended input size(by mean)          |  [592 488] (h x w, multiples of 8)
recommended input size(by mean)          |  [592 496] (h x w, multiples of 16)
recommended input size(by mean)          |  [576 480] (h x w, multiples of 32)

channel mean(0~1)                        |  [0.4954518  0.42574266 0.39330518]
channel std(0~1)                         |  [0.3216056 0.3023355 0.3018837]
*--------------------------------------------------------------------------------------*
```

download site: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  
paper: S. Yang, P. Luo, C. C. Loy, and X. Tang, "From Facial Parts Responses to Face Detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015  

#### Results on [NIH Chest X-ray dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest?hl=ko) (images_001.tar.gz)

<table border="0">
<tr>
    <td>
    <img src="https://user-images.githubusercontent.com/32871371/81142053-6f6ed180-8fa9-11ea-95d4-01412e22d4d5.png" width="100%">
    </td>
    <td>
    <img src="https://user-images.githubusercontent.com/32871371/81142064-7a296680-8fa9-11ea-9940-eb2dc2edcd79.png", width="100%">
    </td>
</tr>
</table>

```
found 4999 images.
Using 12 threads. (max:12)

*--------------------------------------------------------------------------------------*
number of images                         |  4999

dtype                                    |  uint8
channels                                 |  [1, 4]
extensions                               |  ['png']

min height                               |  1024
max height                               |  1024
mean height                              |  1024.0
median height                            |  1024

min width                                |  1024
max width                                |  1024
mean width                               |  1024.0
median width                             |  1024

mean height/width ratio                  |  1.0
median height/width ratio                |  1.0
recommended input size(by mean)          |  [1024 1024] (h x w, multiples of 8)
recommended input size(by mean)          |  [1024 1024] (h x w, multiples of 16)
recommended input size(by mean)          |  [1024 1024] (h x w, multiples of 32)

channel mean(0~1)                        |  [0.5172472 0.5172472 0.5172472]
channel std(0~1)                         |  [0.25274998 0.25274998 0.25274998]
*--------------------------------------------------------------------------------------*
```

data provider: NIH Clinical Center  
download site: https://nihcc.app.box.com/v/ChestXray-NIHCC  
paper: Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald Summers, ChestX-ray8:
Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of
Common Thorax Diseases, IEEE CVPR, pp. 3462-3471, 2017  

### License
[MIT License](https://github.com/Soongja/basic-image-eda/blob/master/LICENSE)
