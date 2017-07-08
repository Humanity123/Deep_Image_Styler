import tensorflow as tf
import numpy as np
import scipy.misc
import VGG19 

class Image:
	def __init__(self, image_matrix=None):
		''' basic constructor of image class based on the pixel values of the image '''
		self.image_matrix = image_matrix

	def get_image_content(self, model, model_type, content_layers):
		''' uses VGG19 model to get the features representing content of an image 
		it is dependent on the model represented to extract these features'''

		if model_type is 'matconvnet':
			#Assumption matconvnet model is only provided, in future may add support for using
			#other saved models 
			graph = tf.Graph()
			with graph.as_default(),tf.Session() as sess:
				vgg_model = VGG19.VGG19()
				model_weights, image_mean_value = vgg_model.read_model_from_matconvnet(model) 
				image = tf.constant(self.image_matrix - image_mean_value, name='input_image')
				vgg_model.get_model_from_matconvnet(input_image, model_weights, pool_type="max")
				image_content = sess.run(content_layers)

			return image_content

	 def get_image_style(self, model, model_type, style_layers):
		''' uses VGG19 model to get the features representing style of an image 
		it is dependent on the model represented to extract these features'''

		if model_type is 'matconvnet':
			#Assumption matconvnet model is only provided, in future may add support for using
			#other saved models
			graph = tf.Graph()
			with graph.as_default, tf.Session() as sess:
				vgg_model = VGG19.VGG19()
				model_weights, image_mean_value = vgg_model.read_model_from_matconvnet(model) 
				vgg_model.get_model_from_matconvnet(input_image, model_weights, pool_type="max")
				image = tf.constant(self.image_matrix - image_mean_value, name='input_image')
				style_layer_values = sess.run(style_layers)
				image_style = []
				for layer in style_layer_values:
					reshaped_layer = np.reshape(layer, [-1, layer.shape[3]])
					gram_matrix = np.matmul(reshaped_layer.T, reshaped_layer)
					image_style.append(gram_matrix)

			return image_style

	def generate_image(model, model_type, content_image, content_layers, style_image, style_layers, content_weight, style_weight, num_iters, feed_back_inters):
		'''it generates the image wose content features matches with the one provides in content_layers
		and style features matches with the one present in the style_layers
		content_weight = alpha
		style_weight = beta'''

		if model_type is 'matconvnet':
			#Assumption matconvnet model is only provided, in future may add support for using
			#other saved models
			content_layer_values = content_image.get_image_content(model, model_type, content_layers)
			style_layer_values   = style_image.get_image_content(model, model_type, style_layers)

			graph = tf.Graph()
			with graph.as_default, tf.Session() as sess:
				image_shape = (1,) + content_image.shape
				input_image = tf.Variable( value = tf.random_normal(shape = image_shape) * 0.256 )

				vgg_model = VGG19.VGG19()
				model_weights, image_mean_value = vgg_model.read_model_from_matconvnet(model)
				vgg_model.get_model_from_matconvnet(input_image, model_weights, pool_type='max')

				content_loss = 0
				for content_layer_index, content_layer in enumerate(content_layers):
					content_loss += tf.nn.l2_loss(vgg_model.layer_dict[content_layer]['tensor_value'] - \
						content_layer_values[content_layer_index])

				style_loss = 0
				style_layer_weight = 1/len(style_layers)
				for style_layer_index, style_layer in enumerate(style_layers):
					style_layer_value_train = vgg_model.layer_dict[style_layer]['tensor_value']
					reshaped_style_layer_train = tf.reshape(style_layer_value_train, [-1, style_layer_value_train.shape[3]])
					gram_matrix = tf.matmul(tf.transpose(reshaped_style_layer_train), reshaped_style_layer_train)
					style_layer_size = style_layer_values[style_layer_index].size
					style_loss += style_layer_weight * tf.nn.l2_loss(gram_matrix - style_layer_values[style_layer_index]) \
					/ (2 * ( style_layer_size ** 2) )

				total_loss = content_weight * content_loss + style_weight * style_loss

				train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(total_loss)
				sess.run(tf.global_variables_initializer())

				minimum_total_loss = total_loss.eval()
				image_with_minimum_total_loss = input_image.eval()

				for iter in range(num_inters):
					train_step.run()

					if iter % feed_back_inters == 0:
						print "Iteration - ", iter
						print "content_loss - ", content_loss.eval()
						print "style_loss - ", style_loss.eval()
						print "total_loss - ", total_loss.eval()

					if total_loss.eval() < minimum_total_loss:
						minimum_total_loss = total_loss.eval()
						image_with_minimum_total_loss = input_image.eval()

				return image_with_minimum_total_loss.reshape(content_image.shape) + image_mean_value

def apply_style():
	''' it applies the style of one image to the other'''
