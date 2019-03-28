This include the training for SDSS/DES dataset. 



* ./benchmarks/  contains the benchmark experiments done on Cooley
* ./data/ contains the datasets needed for training
* ./Xception_final.ipynb is the python notebook used for training on a single GPU
* ./Recursive_Training.ipynb is for recursive training. 
* ./Model_final.py is the python script used for distributed training (using Horovod)
  * splitdata.py is for creating folders for Horovod. Those folders contains symbolic links to the files under data/
  * One could have multiple choices of base models: VGG16, VGG19, InceptionV3, ResNet50, Xception
