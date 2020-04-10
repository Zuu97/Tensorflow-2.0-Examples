import os
data_dir = 'E:\My Projects\Tensorflow 2.0  Basics\Data'
num_epochs = 15
batch_size = 100
kernal_size = (3,3)
pool_size = (2,2)
stride = (2,2)

# # Get Mnist parameters
# conv1 = 32
# conv2 = 64
# conv3 = 128
# dense = 512
# output = 10
# tensor_shape = (28, 28, 1)
# train_path = os.path.join(data_dir,'mnist_train.csv')
# test_path = os.path.join(data_dir,'mnist_test.csv')
# saved_model = os.path.join(data_dir,'mnist_cnn.json')
# saved_weights =  os.path.join(data_dir,'mnist_cnn.h5')

# Get Cifar10 parameters
conv1 = 64
conv2 = 128
conv3 = 256
dense = 512
output = 10
keep_prob = 0.2
tensor_shape = (32, 32, 3)
train_batch_prefix = os.path.join(data_dir,'data_batch_')
test_batch = os.path.join(data_dir,'test_batch')
saved_model = os.path.join(data_dir,'cifar10.json')
saved_weights =  os.path.join(data_dir,'cifar10.h5')
n_batches = 5