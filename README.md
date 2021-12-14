# HAN

Implementation of the HAN network architecture proposed by AUTHORS in https://arxiv.org/abs/2008.08767.

The original network has over 30M parameters. To speed up training and reduce filesizes, the network was built with fewer residual attention blocks.

## How it works

The network was trained using 96x96 patches from the DIV2K dataset. During training for the 2x super resolution, the network uses a 48x48 patch size as input and outputs a 96x96 patch trying to recover the original input.

## Results
Peak signal to noise ratio and structural similarity for the 
x2
|Dataset|Set5|Set14|Urban100|BSDS100|Manga109|
|----|----:|----:|----:|----:|----:|
|PSNR|38.5661|33.0571|32.3999|28.6455|29.9917|
|SSIM|0.9721|0.9266|0.9248|0.8421|0.9444|


x3
|Dataset|Set5|Set14|Urban100|BSDS100|Manga109|
|----|----:|----:|----:|----:|----:|
|PSNR|0|26.2185|24.9875|28.157|25.177|
|SSIM|0|0.7846|0.7578|0.7941|0.8451|



