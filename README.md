# Deep_Image_Styler
An implementation of the paper [Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf) in Tensorflow.

## Parameters

Parameters | Description
-----------|------------
content_image_path| path to image to learn image content
style_image_path | path to image to learn image style
starting_image_path| path to starting image to resume learning 
model | path to the VGG19 model
content_weight | weight/importance given to learning content
style_weight | weight/importance given to learning style
iters | number of iterations

## Results
<p align="center"> 
<img src="/result_images/trans_vid.gif" width="307" height="461">
</p>

### Original Images
Content Image | Style Image
--------------| -----------
![Eiffel Tower](/result_images/tower.jpg) | ![Starry Night](/result_images/night.jpg)

### Generated Images
Starting Image | Content Weight / Style Weight | Final Image
---------------|-------------------------------|------------
![input_image_1_100](/result_images/content_1_style100_input.jpg)|0.01|![output_image_1_100](/result_images/content_1_style_100_output.jpg)
![input_image_1_100](/result_images/content_5_style_1_input.jpg)|5|![output_image_1_100](/result_images/content_5_style_1_output.jpg)
![input_image_1_100](/result_images/content_100_style_1_input.jpg)|100|![output_image_1_100](/result_images/content_100_style_1_output.jpg)
## Requirements

### Pre-trained Models
* [Pre-trained VGG network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) (MD5 8ee3263992981a1d26e73b3ca028a123)

### Dependencies
* TensorFlow
* Pillow
* NumPy
* SciPy


