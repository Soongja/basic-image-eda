# basic-image-eda

A simple eda tool to check basic infos of images under a directory(images are found recursively). This tool was made to quickly check info and prevent mistakes on reading, resizing, and normalizing images as inputs for neural networks. It can be used when first joining an image competition or training CNNs with images!

*Notes:*  
\- All images are converted to 3-channel(rgb) images. When images that have various channels are mixed, results can be misleading.  
\- uint8 and uint16 data types are supported. If different data types are mixed, error occurs.

### Installation
```bash
pip install basic-image-eda
```
prerequisites:
- opencv-python
- numpy
- matplotlib
- tqdm

### Usage(CLI/Code)
#### CLI
simple one line command!
```bash
basic-image-eda <data_dir>
```
or
```bash
basic-image-eda <data_dir> --extensions png jpg --threads 12 --dimension_plot False --channel_hist True --nonzero

Options:
  -e --extensions        target image extensions.(default=['png', 'jpg', 'jpeg'])
  -t --threads           number of multiprocessing threads. if zero, automatically counted.(default=0)
  -d --dimension_plot    show dimension(height/width) scatter plot.(default=True)
  -c --channel_hist      show channelwise pixel value histogram. takes much longer time.(default=False)
  -n --nonzero           calculate values only from non-zero pixels of the images.(default=False)
  -V --version           show version.
```

#### Code
```python
from basic_image_eda import BasicImageEDA

if __name__ == "__main__":  # for multiprocessing
    data_dir = "./data"

    # below are default values. 
    extensions = ['png', 'jpg', 'jpeg']
    threads = 0
    dimension_plot = True
    channel_hist = False
    nonzero = False

    BasicImageEDA.explore(data_dir, extensions, threads, dimension_plot, channel_hist, nonzero)
```

### Results
#### Results on [celeba dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (test set)

<table border="0">
<tr>
    <td>
    <img src="https://user-images.githubusercontent.com/32871371/74670364-6103eb80-51ec-11ea-9196-94e042820d0c.png" width="100%">
    </td>
    <td>
    <img src="https://user-images.githubusercontent.com/32871371/74670511-ae805880-51ec-11ea-802b-63d0b699fa52.png", width="100%">
    </td>
</tr>
</table>

```
found 19962 images.
Using 12 threads.

*--------------------------------------------------------------------------------------*
number of images          |  19962

dtype                     |  uint8
channels                  |  [3]
extensions                |  ['jpg']

min height                |  85
mean height               |  591.8215108706543
max height                |  5616

min width                 |  85
mean width                |  490.2976655645727
max width                 |  5616

mean height/width ratio   |  1.207065732587525
recommended input size    |  [592 488] (h x w, multiples of 8)
recommended input size    |  [592 496] (h x w, multiples of 16)
recommended input size    |  [576 480] (h x w, multiples of 32)

channel mean(0~1)         |  [0.49546506 0.42573904 0.39331011]
channel std(0~1)          |  [0.32161251 0.30237885 0.30192492]
*--------------------------------------------------------------------------------------*
```

#### Results on [NIH Chest X-ray dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest?hl=ko) (images_001.tar.gz)

<table border="0">
<tr>
    <td>
    <img src="https://user-images.githubusercontent.com/32871371/74670529-b93aed80-51ec-11ea-8aff-f1e2bcbcf622.png" width="100%">
    </td>
    <td>
    <img src="https://user-images.githubusercontent.com/32871371/74670548-c3f58280-51ec-11ea-8fff-4a7006053415.png", width="100%">
    </td>
</tr>
</table>

```
found 4999 images.
Using 12 threads.

*--------------------------------------------------------------------------------------*
number of images          |  4999

dtype                     |  uint8
channels                  |  [1, 4]
extensions                |  ['png']

min height                |  1024
mean height               |  1024.0
max height                |  1024

min width                 |  1024
mean width                |  1024.0
max width                 |  1024

mean height/width ratio   |  1.0
recommended input size    |  [1024 1024] (h x w, multiples of 8)
recommended input size    |  [1024 1024] (h x w, multiples of 16)
recommended input size    |  [1024 1024] (h x w, multiples of 32)

channel mean(0~1)         |  [0.51725466 0.51725466 0.51725466]
channel std(0~1)          |  [0.25274113 0.25274113 0.25274113]
*--------------------------------------------------------------------------------------*
```

### License
[MIT License](https://github.com/Sonngja/basic-image-eda/blob/master/LICENSE)
