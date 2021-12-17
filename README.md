# HAN

Implementation of the HAN-like network architecture proposed by AUTHORS in https://arxiv.org/abs/2008.08767. To speed up training/evaluation, the network was built with fewer residual attention blocks (64 instead of 200 residual channel attention blocks).

## How it works

The network was trained using 96x96 patches from the DIV2K dataset. During training, the input is downsampled by the scale factor, the fed in as a lower resolution image. To virtually increase the size of the dataset, the images are randomly horizontally flipped and rotated.

## Results
Peak signal to noise ratio and structural similarity for the 
x2
|Dataset|Set5|Set14|Urban100|BSDS100|Manga109|
|----|----:|----:|----:|----:|----:|
|PSNR|38.5661|33.0571|32.3992|33.6106|36.6606|
|SSIM|0.9721|0.9266|0.9248|0.9187|0.9757|


x3
|Dataset|Set5|Set14|Urban100|BSDS100|Manga109|
|----|----:|----:|----:|----:|----:|
|PSNR|0|26.2185|24.9875|28.1571|25.1775|
|SSIM|0|0.7846|0.7578|0.7941|0.8451|

![Super Resolution x2](SR_x2.jpg)





