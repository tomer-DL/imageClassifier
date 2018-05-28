# Training a neural network to distinguish faces from non faces images

In this part we will run our dataset through a Convolutional Neural Network (CNN), to learn which images are faces and which are not.

### set the properties in part3.ini (or leave them with default values):

* dataset.dir= The directory where the dataset you created in part2 is.
* dataset.file.name.x= The file name of the file containing the images
* dataset.file.name.y= The file name of the file containing the labels for the images (1 for face, 0 for non-face)
* model.save.dir= The directory to save the model after training in order to use it later on
* model.save.name= The model file name
* real.faces.dir= The faces images directory
* not.real.faces.dir= The non faces images directory

Then, run **face_or_not.py**
