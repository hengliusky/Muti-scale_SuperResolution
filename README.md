How to run:

	NVIDIA GPU + CUDA CuDNN 
	Caffe
	The respective training code and test code for a single edge loss training, single image training, and multiple loss training are included in 
	./caffe_file and ./test.(For example, we want to train with a single image loss function and test it. Then we select the training code and parameter 
	configuration from ./caffe_file/Rect/* and select the test network and test code from ./test/Rect/*.)

Training:

	Rotate, scale, and flip the image in ./data/91 images.
	Make the train image into hdf5 file with ./matlab/generate_train.m，and make the test image into hdf5 with ./matlab/generate_train.m
	The network structure and configuration parameters are in ./caffe_files，and select the loss function to select the corresponding training code (Solver 
	is the configuration file, and deploy is the network structure.)
	for training.
	./model contains the network model trained by the respective loss function, which can be used for later testing.
	
Test：

	The benchmark image of the test is stored in ./data/Test.
	The test code and test model for each loss function training are stored in their respective folders in ./test.
	You can select the corresponding model from the ./model folder for testing, or you can get the network model to 
	test by training yourself.
	
	
