## Face classifier using open CV and deep learning

This is an educational project aimed at studying face recognition.
It is devided into 10 different parts.
Each part has it's own directory with independant code. 
For each part there is an *ini* file specifying its parameters, and a README.md file explaining how it works, and how to config those parameters. Most of the parts are based on previous parts, so it is recommended that you follow their order.

This is a short explaination of each part. Each part has a more detailed explaination in its own README.md 

* In the **1st part** I am gathering face images dataset by using images saved on my computer.
I am using Haar Cascade to recognize faces inside images, and extract those faces to create a faces dataset. This method is not totally accurate, so we would have to address the issue in the next few parts. 

* In the **2nd part**, since the Haar Cascade isn't totally accurate, I am creating a dataset to distinguish genuine face images from false classifications. For this you would have to go through your faces photos and create two different directories. The first will hold genuine face images. The second will hold images that were mistakenly classifed as faces. After creating these two directories, running the code will create a dataset, to be used to train a "face or not" classifier.

* In the **3rd part**, since the Haar Cascade isn't totally accurate, I am training a deep neural network to distinguish
between true face images and false face images.

* In the **4th part** I am using the deep neural network created in the third part to sort the rest of the faces dataset. The sort would not be perfect, but it eases a lot the manual sort.

* For the **5th part** After manually sorting all the dataset, I am training a deep neural network on the full sorted dataset. Since the dataset is too big to hold in the memory, unlike the one created in the 2nd part, I am using a generator to create the dataset as I am training it.
