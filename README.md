# Single Image Super-Resolution Using Multi-Scale Deep Encoder-Decoder with Phase Congruency Edge Map Guidance

## 1.Abstract
<br>This paper presents an end-to-end multi-scale deep encoder (convolution) and
decoder (deconvolution) network for single image super-resolution (SISR)
guided by phase congruency (PC) edge map. Our system starts by a single
scale symmetrical encoder-decoder structure for SISR, which is extended to
a multi-scale model by integrating wavelet multi-resolution analysis into our
network. The new multi-scale deep learning system allows the low resolution
(LR) input and its PC edge map to be combined so as to precisely predict
the multi-scale super-resolved edge details with the guidance of the high-resolution (HR) PC edge map. In this way, the proposed deep model takes
both the reconstruction of image pixels’ intensities and the recovery of multi-scale edge details into consideration under the same framework. We evaluate
the proposed model on benchmark datasets of different data scenarios, such
as Set14 and BSD100 - natural images, Middlebury and New Tsukuba -
depth images. The evaluations based on both PSNR and visual perception
reveal that the proposed model is superior to the state-of-the-art methods.</br>

## 2. The proposed multi-scale deep network
<br>The detailed architecture of the MSDEPC network is shown below.</br>
![image](https://github.com/hengliusky/Muti-scale-SuperResolution/blob/master/imgs/Net.png)
<br>The proposed MSDEPC model: joint input of LR image and edge (red), multi-scale encoder-decoder learning (yellow), multi-scale HR image and edge prediction (green),
and the total loss (blue).</br>

<br>The structure of the single-scale encode-decoder is shown in below</br>
![image](https://github.com/hengliusky/Muti-scale-SuperResolution/blob/master/imgs/encode-decode.png)
<br>Single scale deep symmetrical encoder-decoder: the network only consists of
convolutional and deconvolutional layers; PReLU layer is followed after convolution or
deconvolution operation.</br>

## 3.Usage

### 1)Requirements
<br>Ubuntu 14.04 LTS</br>
<br>MATLAB R2014a</br>
<br>CUDA 8.0 and Cudnn 5.1</br>
<br>Caffe</br>
<br>Nvidia TITAN X</br>

### 2)Dataset and data processing
**Dataset**
Our network utilzes two different dataset to train the network:
 <br>① 91-images Enhance the 91-images dataset to train multi-scale deep network </br>
 <br>② ImageNet Extract 50,000 images from imagenet to train  multi-scale deep network </br>
**Data enhancement**
 <br>./matlab/data augment.m Enhances the image, including rotation, scaling, and flipping.</br>
**Data generation**
 <br>./matlab/generate_train.m Convert training images to HDF5 files</br>
 <br>./matlab/generate_train.m Convert testing images to HDF5 files</br>

### 3）Training
<br>**Training multiple loss networks**</br>
<br> /path_to_caffe/build/tools/caffe train -solver ./caffe_file/Mutiloss/mutiscale_solver.prototxt -gpu [gpu id]</br>
<br>**Training a single Edge loss network**</br>
<br> /path_to_caffe/build/tools/caffe train -solver ./caffe_file/Edge/mutiscale_solver_edge1.prototxt -gpu [gpu id]</br> 
<br>**Training a single image loss network**</br>
<br> /path_to_caffe/build/tools/caffe train -solver ./caffe_file/Rect/mutiscale_solver_rect2.prototxt -gpu [gpu id]</br>

### Model
**Model obtained through multiple loss training**
<br>./model/mutiscale_iter_223500.caffemodel</br>

**Model obtained through edge loss training**
<br>./model/mutiscale_edge1_iter_154500.caffemodel</br>

**Model obtained through image loss training**
<br>./model/mutiscale_rect2_iter_205000.caffemodel</br>

### Test
<br>The benchmark test images included the BSD100, Set 5 and Set 14 data sets.</br>
**Test network performance based on multiple losses**
<br>Test code: ./test/Mutiloss/mutiscale3_val.m</br>
<br>Test network: ./test/Mutiloss/test.prototxt</br>

**Test network performance based on edge losses**
<br>Test code: ./test/Edge/test_edge.m</br>
<br>Test network: ./test/Edge/test_edge1.prototxt</br>

**Test network performance based on multiple losses**
<br>Test code: ./test/Rect/test_rect.m</br>
<br>Test network: ./test/Rect/test_rect1.prototxt</br>
<br>You can select the corresponding model from the ./model folder for testing, or you can get the network model to 
test by training yourself.</br>

### Comparison standard
<br>We convert RGB images to YCBCR images and only use the Y channel for training and testing.</br>
<br>The PSNR and SSIM of the image are calculated by comparing the Y channel of the reconstructed image with the Y channel of the original image. The higher the value, the higher the quality of the image reconstruction.</br>

### Performance comparison
![image](https://github.com/hengliusky/Muti-scale-SuperResolution/blob/master/imgs/result1.png)
![image](https://github.com/hengliusky/Muti-scale-SuperResolution/blob/master/imgs/result2.png)
Comparison of visual and PSNR for images '148026' and '106024' from BSD100 by (a) SRCNN-L, (b) VDSR, and (c) the
proposed - MSDEPC, respectively.

![image](https://github.com/hengliusky/Muti-scale-SuperResolution/blob/master/imgs/table.png)
The mean PSNR (dB) (left numbers) and SSIM (right numbers) for different
methods trained with 91-images. Best results are indicated in Bold.
