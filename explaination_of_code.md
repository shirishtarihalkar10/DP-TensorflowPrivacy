Now we will train a Convolutional neural network on MNIST database which is like the hello world of Deep Learning using Keras and the Differentially private stochastic gradient descent optimizer.

First we import from future absolute_import , division and print_function.

Then from abseil lib we import app ,flags and logging which will help us in making the model.

We also import numpy and tensorflow.

Then from tensorflow privacy we import the compute_rdp , get_privacy_spent and DPsequential.

We then define some flags using flags which we imported from abseil.

We set DPSGD to true , learning rate to 0.15 , noise_multipler to 0.1 , L2_norm_clip to 1.0 , batch_size to 250 , epochs to 60 and microbatches to 250.

Now I will explain some of the parameters we defned here

Learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model is updated. The higher the learning rate, the more each update matters.If the updates are noisy (such as when the additive noise is large compared to the clipping threshold), the learning rate must be kept low for the training procedure to converge.

Next is noise_multipler 

Noise Multiplier governs the amount of noise added during training. Generally, more noise results in better privacy and lower utility. This generally has to be at least 0.3 to obtain rigorous privacy guarantees, but smaller values may still be acceptable for practical purposes.

l2_norm_clip (float): The cumulative gradient across all network parameters from each microbatch will be clipped so that its L2 norm is at most this value. We should set this to something close to some percentile of what you expect the gradient from each microbatch to be.

The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.

The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. 

microbatches (int): The input data for each step (i.e., batch) of our original training algorithm is split into this many microbatches. Generally, increasing this will improve our utility but slow down our training in terms of wall-clock time. The total number of examples consumed in one global step remains the same. This number should evenly divide our input batch size.

Then we define a function for computing epsilon.

Epsilon measures the strength of our privacy guarantee. In the case of differentially private machine learning, it gives a bound on how much the probability of a particular model output can vary by including (or removing) a single training example. We usually want it to be a small constant. However, this is only an upper bound, and a large value of epsilon could still mean good practical privacy.

First, we need to define a list of orders, at which the Rényi divergence will be computed. The first method compute_rdp returns the Rényi differential privacy achieved by the Gaussian mechanism applied to gradients in DP-SGD, for each of the orders.

Then, the method get_privacy_spent computes the best epsilon for a given target_delta value of delta by taking the minimum over all orders.

Delta bounds the probability of our privacy guarantee not holding. A rule of thumb is to set it to be less than the inverse of the training data size (i.e., the population size). 

Running that function with the hyperparameter values used during training will estimate the epsilon value that was achieved by the differentially private optimizer, and thus the strength of the privacy guarantee which comes with the model we trained. 

Then we define a function to load the MNIST data using keras.

We use keras.dataset.mnist.load_data() to load the data into train and test.

It returns two tuples of numPy arrays.We store the data in train_data , test_data and labels in train_labels and test_labels respectively.

The train data is used to train the model whereas the test data is used to test the model.

Once we have loaded the data we Preprocess the data 

We use numPy array on the data , specify the desired data type of the array as float32 and divide the array by 255.

Then reshape the data to desired format.

We then apply numpy array to train labels and then call the keras.utils to_categorical function which converts the class vector into a binary matrix of total number of classes to 10.

Then we check whether the min of train data and test data is 0 and max is 1.

If not it raises an Assertion Error because we have use the assert keyword.

We then return the data respectivily.

In the main function we set logging verbosity to INFO which means tensorflow will display all the messages that have the label INFO.

We then check whether the number of microbatches divide the batch size evenly.

Call the load_mnist function to load the data.

Then we define the sequential model layers which we are gonna apply. 

We first apply the 2D convolution layer and pass on the required arguments.

We need 16 output filters ,
 kernel size as 8 , 
strides as 2, 
padding as "same" which means padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input,
activation function as rectified linear unit activation function.

We then apply the Max pooling operation for 2D spatial data and pass the respective arguments.

We pass the pool size as 2 and strides as 1.

After that we again apply the 2D convolution layer but this time we need 32 output filters and kernel size as 4.

We apply the Max pooling layer again and then apply the flatten layer which flatten the input.

We then apply the dense layer and pass the units as 32 and activation function as rectified linear unit activation function.

We apply the dense layer again and pass units argument as 10.

Next we call the Diffrential Privacy sequential function which we had imported earlier and pass in the arguments l2_norm_clip , noise_multiplier and layers which we have defined before .

Then we call the gradient desect model and pass the learning rate which we specifed earlier.

For calculating the loss we use keras CategoricalCrossentropy which computes the crossentropy loss between the label and predicton.

We then train the model using model.fit and pass the training data , traing labels , epochs ,batch size and validation data.

Then atlast we display the epsilon value using the compute epsilon function we defined earlier.



