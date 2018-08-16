# Single Image Super-Resolution Using Multi-Scale Deep Encoder-Decoder with Phase Congruency Edge Map Guidance

## Abstract
<br>This paper presents an end-to-end multi-scale deep encoder (convolution) and
decoder (deconvolution) network for single image super-resolution (SISR)
guided by phase congruency (PC) edge map. Our system starts by a single
scale symmetrical encoder-decoder structure for SISR, which is extended to
a multi-scale model by integrating wavelet multi-resolution analysis into our
network. The new multi-scale deep learning system allows the low resolution
(LR) input and its PC edge map to be combined so as to precisely predict
the multi-scale super-resolved edge details with the guidance of the highresolution (HR) PC edge map. In this way, the proposed deep model takes
both the reconstruction of image pixels’ intensities and the recovery of multiscale edge details into consideration under the same framework. We evaluate
the proposed model on benchmark datasets of different data scenarios, such
as Set14 and BSD100 - natural images, Middlebury and New Tsukuba -
depth images. The evaluations based on both PSNR and visual perception
reveal that the proposed model is superior to the state-of-the-art methods.</br>
