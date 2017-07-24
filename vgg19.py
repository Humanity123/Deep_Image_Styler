import tensorflow as tf
import numpy as np
import scipy.io

# Defining the basic structure of a layer
def get_layer_struct(layer_type, layer_name):
	''' return a normal struct with the type and the name of the layer'''
	layer = {'layer_type':layer_type, 'layer_name':layer_name, 'tensor_value': None}
	return layer

class VGG19:
	def __init__(self):
		self.VGG19_Layers = (
			get_layer_struct('conv', 'conv1_1'),get_layer_struct('relu', 'relu1_1'),get_layer_struct('conv', 'conv1_2'),get_layer_struct('relu', 'relu1_2'),
			get_layer_struct('pool', 'pool1_1'),

			get_layer_struct('conv', 'conv2_1'),get_layer_struct('relu', 'relu2_1'),get_layer_struct('conv', 'conv2_2'),get_layer_struct('relu', 'relu2_2'),
			get_layer_struct('pool', 'pool2_1'),

			get_layer_struct('conv', 'conv3_1'),get_layer_struct('relu', 'relu3_1'),get_layer_struct('conv', 'conv3_2'),get_layer_struct('relu', 'relu3_2'),
			get_layer_struct('conv', 'conv3_3'),get_layer_struct('relu', 'relu3_3'),get_layer_struct('conv', 'conv3_4'),get_layer_struct('relu', 'relu3_4'),
			get_layer_struct('pool', 'pool3_1'),

			get_layer_struct('conv', 'conv4_1'),get_layer_struct('relu', 'relu4_1'),get_layer_struct('conv', 'conv4_2'),get_layer_struct('relu', 'relu4_2'),
			get_layer_struct('conv', 'conv4_3'),get_layer_struct('relu', 'relu4_3'),get_layer_struct('conv', 'conv4_4'),get_layer_struct('relu', 'relu4_4'),
			get_layer_struct('pool', 'pool4_1'),

			get_layer_struct('conv', 'conv5_1'),get_layer_struct('relu', 'relu5_1'),get_layer_struct('conv', 'conv5_2'),get_layer_struct('relu', 'relu5_2'),
			get_layer_struct('conv', 'conv5_3'),get_layer_struct('relu', 'relu5_3'),get_layer_struct('conv', 'conv5_4'),get_layer_struct('relu', 'relu5_4')
		)
		
		self.layer_dict = {}
		for layer in self.VGG19_Layers:
			self.layer_dict[layer['layer_name']] =layer

	def read_model_from_matconvnet(self, saved_model):
		'''reads the saved matconvnet model and returns the weight and the mean pixel value'''
		matconvnet_model = scipy.io.loadmat(saved_model)
		mean_value = np.mean(matconvnet_model['normalization'][0][0][0])
		weights    = matconvnet_model['layers'][0]
		return weights, mean_value

	def get_model_from_matconvnet(self, input, weights, pool_type):
		''' returns a trained VGG19 model with weights taken from a pre-trained matconvnet model'''
		input_to_next_layer = input
		for layer_index, layer in enumerate(self.VGG19_Layers):
			if layer['layer_type'] == "conv":
				conv_filter_weight, conv_filter_bias = weights[layer_index][0][0][0][0] 
				rearrange = (1,0,2,3)
				conv_filter_weight = np.transpose(conv_filter_weight, rearrange)
				conv_filter_bias = conv_filter_bias.reshape(-1)
				conv_layer = tf.nn.conv2d(input=input_to_next_layer, filter=tf.constant(conv_filter_weight), strides=[1,1,1,1], padding='SAME')
				input_to_next_layer = tf.nn.bias_add(conv_layer, conv_filter_bias)
			elif layer['layer_type'] == "relu":
				input_to_next_layer = tf.nn.relu(input_to_next_layer)
			elif layer['layer_type'] == "pool":
				if pool_type == "avg":
					input_to_next_layer = tf.nn.avg_pool(value=input_to_next_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
				elif pool_type == "max":
					input_to_next_layer = tf.nn.max_pool(value=input_to_next_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

			layer['tensor_value'] = input_to_next_layer
