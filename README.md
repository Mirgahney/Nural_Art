# Nural_Art
Implementation of Artistic Style Transfer with Artificial Neural Network.

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
