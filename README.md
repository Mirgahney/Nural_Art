# Nural_Art
Implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

In this paper, style transfer uses the features found in the 19-layer [VGG Network](https://arxiv.org/pdf/1409.1556.pdf), which is comprised of a series of convolutional and pooling layers, and a few fully-connected layers. In the image below, the convolutional layers are named by stack and their order in the stack. <i>Conv_1_1</i> is the first convolutional layer that an image is passed through, in the first stack. <i>Conv_2_1</i> is the first convolutional layer in the second stack. The deepest convolutional layer in the network is <i>conv_5_4</i>.

## Requirements :
* Torch >= 0.04
* Torchvision >= 0.2.1

## To run the code:
```
python generate_images.py
```

The code will reads the images from Images folder and the result will be saved in the same folder under name result.

## Sample of the result:
![Octopus with delaunay](/Images/octopus_result.jpg)
![Space with the starry night](/Images/space_needle2_lgram.jpg)
![Space with screem](/Images/space_needle_lgram.jpg)
The result obtained by running the code with <i>content_weight = 1 and style_weight = 1e6</i> on several content images and with a famous style paints.    
