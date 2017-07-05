import tensorflow as tf
import numpy as np
import VGG19 

class Image:
	def __init__(self, image_matrix):
		''' basic constructor of image class based on the pixel values of the image '''
		self.image_matrix = image_matrix

	def get_image_content(self, model=None, content_layers):
		''' uses VGG19 model to get the features representing content of an image 
		it is dependent on the model represented to extract these features'''
		if model is not None:
			#Assumption matconvnet model is only provided, in future may add support for using
			#default saved models in case no external input model is provided
			with tf.Session() as sess:
				image = tf.constant(self.image_matrix, name='input_image')
				vgg_model = VGG19.VGG19()
				model_weights, image_mean_value = vgg_model.read_model_from_matconvnet(model) 
				vgg_model.get_model_from_matconvnet(input_image, model_weights, pool_type="max")
				image_content = sess.run(content_layers)

			return image_content

	 def get_image_style(self, model=None, style_layers):
		''' uses VGG19 model to get the features representing style of an image 
		it is dependent on the model represented to extract these features'''
		if model is not None:
			#Assumption matconvnet model is only provided, in future may add support for using
			#default saved models in case no external input model is provided
			with tf.Session() as sess:
				image = tf.constant(self.image_matrix, name='input_image')
				vgg_model = VGG19.VGG19()
				model_weights, image_mean_value = vgg_model.read_model_from_matconvnet(model) 
				vgg_model.get_model_from_matconvnet(input_image, model_weights, pool_type="max")
				image_style = sess.run(style_layers)

			return image_style






def apply_style():
	''' it applies the style of one image to the other'''
