from   scipy.misc import imread
import PIL 
import os
import tensorflow as tf
import numpy as np
import vgg19 

class Image:
	def __init__(self, image_matrix=None, image_path=None):
		''' basic constructor of image class'''
		if image_path is not None:
			self.image_matrix = self.read_image_from_path(image_path)
		else:
			self.image_matrix = image_matrix

	def save_image(self, image_name, image_path=None):
		if image_path is None:
			image_path = os.path.join(os.getcwd(), image_name+".jpg")
		clipped_image_matrix = np.clip(self.image_matrix, 0, 255).astype(np.uint8)
		PIL.Image.fromarray( clipped_image_matrix ).save(image_path, quality=95)

	def read_image_from_path(self, image_path):
		image_matrix = imread(image_path).astype(np.float32)

		#Method For GrayScale Images
		if len(image_matrix.shape) == 2:
			image_matrix = np.dstack(image_matrix, image_matrix, image_matrix)

		#Method for PNG images with 4  channels:
		if image_matrix.shape[2] == 4:
			image_matrix = image_matrix[:, :, :3]

		return image_matrix

	def get_image_content(self, model, model_type, content_layers):
		''' uses VGG19 model to get the features representing content of an image 
		it is dependent on the model represented to extract these features'''

		if model_type is 'matconvnet':
			#Assumption matconvnet model is only provided, in future may add support for using
			#other saved models 
			graph = tf.Graph()
			with graph.as_default(), tf.Session() as sess:
				vgg_model = vgg19.VGG19()
				model_weights, image_mean_value = vgg_model.read_model_from_matconvnet(model) 
				input_image = tf.constant(np.reshape(self.image_matrix - image_mean_value, (1,) + self.image_matrix.shape)\
					, name='input_image')
				vgg_model.get_model_from_matconvnet(input_image, model_weights, pool_type="max")
				tensor_list_content_layers = [vgg_model.layer_dict[layer]['tensor_value'] for layer in content_layers]
				image_content = sess.run(tensor_list_content_layers)		

			return image_content

	def get_image_style(self, model, model_type, style_layers):
		''' uses VGG19 model to get the features representing style of an image 
		it is dependent on the model represented to extract these features'''

		if model_type is 'matconvnet':
			#Assumption matconvnet model is only provided, in future may add support for using
			#other saved models
			graph = tf.Graph()
			with graph.as_default(), tf.Session() as sess:
				vgg_model = vgg19.VGG19()
				model_weights, image_mean_value = vgg_model.read_model_from_matconvnet(model) 
				input_image = tf.constant(np.reshape(self.image_matrix - image_mean_value, (1,) + self.image_matrix.shape)\
					, name='input_image')
				vgg_model.get_model_from_matconvnet(input_image, model_weights, pool_type="max")
				tensor_list_style_layers = [vgg_model.layer_dict[layer]['tensor_value'] for layer in style_layers]
				style_layer_values = sess.run(tensor_list_style_layers)
				image_style = []
				for layer in style_layer_values:
					reshaped_layer = np.reshape(layer, [-1, layer.shape[3]])
					gram_matrix = np.matmul(reshaped_layer.T, reshaped_layer)
					image_style.append(gram_matrix)

			return image_style

	def generate_image(self, model, model_type, content_image, content_layers, style_image, style_layers, content_weight, style_weight, num_iters, feed_back_inters, starting_image = None):
		'''it generates the image wose content features matches with the one provides in content_layers
		and style features matches with the one present in the style_layers
		content_weight = alpha
		style_weight = beta'''

		if model_type is 'matconvnet':
			#Assumption matconvnet model is only provided, in future may add support for using
			#other saved models
			content_layer_values = content_image.get_image_content(model, model_type, content_layers)
			style_layer_values   = style_image.get_image_style(model, model_type, style_layers)

			print len(style_layer_values)
			for i in style_layer_values:
				print i.shape

			graph = tf.Graph()
			with graph.as_default(), tf.Session() as sess:
				vgg_model = vgg19.VGG19()
				model_weights, image_mean_value = vgg_model.read_model_from_matconvnet(model)

				image_shape = (1,) + content_image.image_matrix.shape
				if starting_image is None:
					input_image = tf.Variable(tf.random_normal(shape = image_shape) * 0.256 )
				else:
					input_image = tf.Variable( np.reshape(starting_image.image_matrix - image_mean_value, image_shape) )

				vgg_model.get_model_from_matconvnet(input_image, model_weights, pool_type='max')

				content_loss = 0
				for content_layer_index, content_layer in enumerate(content_layers):
					content_layer_size = content_layer_values[content_layer_index].size
					content_loss += tf.nn.l2_loss(vgg_model.layer_dict[content_layer]['tensor_value'] - \
						content_layer_values[content_layer_index]) 

				style_loss = 0
				style_layer_weight = 1.0/len(style_layers)
				for style_layer_index, style_layer in enumerate(style_layers):
					style_layer_value_train = vgg_model.layer_dict[style_layer]['tensor_value']
					print style_layer_value_train.get_shape()[3].value
					reshaped_style_layer_train = tf.reshape(style_layer_value_train, (-1, style_layer_value_train.get_shape()[3].value	))
					gram_matrix = tf.matmul(tf.transpose(reshaped_style_layer_train), reshaped_style_layer_train)
					style_layer_size = style_layer_values[style_layer_index].size
					style_loss += style_layer_weight * tf.nn.l2_loss( (gram_matrix - style_layer_values[style_layer_index]) / style_layer_size ) \
					/ 2 

				total_loss = content_weight * content_loss + style_weight * style_loss

				learning_rate = 1e1
				beta1 = 0.9
				beta2 = 0.999
				epsilon = 1e-08	
				train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(total_loss)
				sess.run(tf.global_variables_initializer())

				minimum_total_loss = total_loss.eval()
				image_with_minimum_total_loss = input_image.eval()

				for iter in range(num_iters):
					#train_step.run()
					content_loss_value, style_loss_value, total_loss_value, _ = sess.run([content_loss, style_loss, total_loss, train_step])
					if iter%10 == 0:
						save_image = Image(image_with_minimum_total_loss.reshape(content_image.image_matrix.shape) + image_mean_value)
						save_image.save_image(image_name = str(iter))

					if iter % feed_back_inters == 0:
						print "Iteration - ", iter
						print "content_loss - ", content_loss_value
						print "style_loss - ", style_loss_value
						print "total_loss - ", total_loss_value

					if total_loss_value < minimum_total_loss:
						minimum_total_loss = total_loss_value
						image_with_minimum_total_loss = input_image.eval()

				return image_with_minimum_total_loss.reshape(content_image.image_matrix.shape) + image_mean_value

def apply_style(model, model_type, content_image_path, style_image_path, starting_image_path = None, content_weight = 1, style_weight = 1, iters = 1000):
	''' it applies the style of one image to the other'''
	content_layers = ['relu1_1']
	style_layers = ['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1']
	content_image = Image(image_path = content_image_path)
	style_image   = Image(image_path = style_image_path)
	if starting_image_path is not None:
		starting_image = Image(image_path = starting_image_path)
	
	generated_image = Image(content_image.generate_image(model, 'matconvnet', content_image,content_layers, style_image,style_layers,content_weight, style_weight, iters, 1, starting_image))
	generated_image.save_image(image_name='final')



def main():
	content_image_path = ''
	style_image_path = ''
	starting_image_path  = ''
	model = ''
	content_weight = 1
	style_weight = 1
	iters = 1000
	apply_style(model, 'matconvnet', content_image_path, style_image_path, starting_image_path, content_weight, style_weight, iters)


if __name__ == "__main__":
	main()
