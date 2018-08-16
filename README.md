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

## 2. The network structure we proposed
<br>The detailed architecture of the MSDEPC network is shown below.</br>

<br>The proposed MSDEPC model: joint input of LR image and edge (red), multi-scale encoder-decoder learning (yellow), multi-scale HR image and edge prediction (green),
and the total loss (blue).</br>

<br>The structure of the encode-decoder is shown below</br>

<br>Single scale deep symmetrical encoder-decoder: the network only consists of
convolutional and deconvolutional layers; PReLU layer is followed after convolution or
deconvolution operation.</br>

## 3.Usage

### Requirements
<br>MATLAB R2014a</br>
<br>CUDA 8.0 and Cudnn 5.1</br>
<br>Caffe</br>

### Computer configuration
<br>Ubuntu 14.04 LTS</br>
<br>Intel Xeon E5-2620</br>
<br>Memory 64GB</br>
<br>Nvidia GeForce GTX 1080TI</br>

### Data enhancement and generation
**Data enhancement**
 ./matlab/data augment.m Enhances the image, including rotation, scaling, and flipping.
**Data generation**
 ./matlab/generate_train.m Convert training images to HDF5 files
 ./matlab/generate_train.m Convert testing images to HDF5 files
