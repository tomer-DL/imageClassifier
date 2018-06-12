## Part 1

In the 1st part you will be gathering face images dataset by using images saved on your computer. 
You will be using Haar Cascade to recognize faces inside images, and extract those faces to create a faces dataset. 
This method is not totally accurate, so we would have to address the issue in the next few parts.

This part includes the following files:

* part1.ini - includes the parameters for creating the dataset:
  * face.classifier=haarcascade_frontalface_default.xml - this is a haar cascade file taken from  [opencv repository](https://github.com/opencv/opencv/tree/master/data/haarcascades)
  * input.images.root - this is the root directory of your images. the images will be read recursively
  * images.extension - the input images extension (like .jpg)
  * factor.x=0.9 - the horizontal sizing factor of the images read. A value between 0.2 and 1. A bigger number means more faces, but more strain on your computer memory during the creation of the dataset.
  * factor.y=0.9 - the verticalal sizing factor of the images read.
  * face.min.size=150 the size of the output face images. A bigger face would be resized to 150X150. A smaller one would be ignored.
  * output.faces.dir - the directory to save the face images to
  * faces.extension=.jpg  - the file format to save the face images to
* utils.py - Utility functions to read an ini file to a dictionary, and to list all files in a directory
* **file-generator.py** - This is the program to run. It reads recursively all the images from the *input.images.root* directory, extract the faces, resizes them to *face.min.sizeXface.min.size*, and saves them to the *output.faces.dir* directory.

After this part you should have a dataset of mostly face images. In my case, I had about 10,000 dataset images, with about 89% of them genuine face images.
