# Creating a small dataset for the initial deep learning

Deep learning works best on a big dataset. In my case, after the first part, I had about 10,000 images.
Sorting through those images can be a tedious job.
So instead of sorting all of those images, I decided to create a smaller sample. 
Then, run it through the neural network, and let the deep learning do the initial sorting.
In this part we are creating the small dataset.
In the next part we'll train and then sort the remaining files.

For this part, you should take 200 genuine face images and put them in one directory.
then take 200 non face images and put them in another directory.

### set the properties in part2.ini (or leave them with default values):
* input.images.faces= The faces images directory
* input.images.other= The non faces images directory
* images.extension= The faces images extension (i.e. .jpg)
* face.size.height= The width of the images (150 if you didn't change the default in part1)
* face.size.width= The height of the images (150 if you didn't change the default in part1)
* output.dataset.dir= The output directory for the dataset files
* dataset.file.name.x= The file name of the file containing the images
* dataset.file.name.y= The file name of the file containing the labels for the images (1 for face, 0 for non-face)

Then, run **create_dataset.py**
