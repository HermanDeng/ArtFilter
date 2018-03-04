import numpy as np
import scipy.io
import tensorflow as tf
import utils

VGG_WEIGHTS = 'imagenet-vgg-verydeep-19.mat'

class VGGnet(object):
	def __init__(self, input_imag):
		self.vggnet = scipy.io.loadmat(VGG_WEIGHTS)['layers']
		self.input_img = input_imag
		self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


	def get_weights(self, layer_idx):
		W = self.vggnet[0][layer_idx][0][0][2][0][0]
		b = self.vggnet[0][layer_idx][0][0][2][0][1]
		# flatten b to 1D array
		return W, b.reshape(b.size)


	def conv2d_relu(self, prev_layer, layer_idx, layer_name):
		with tf.variable_scope(layer_name) as scope:
			w, b = self.get_weights(layer_idx)
			weights = tf.constant(w, name = 'weights')
			bias = tf.constant(b, name = 'bias')
			conv = tf.nn.conv2d(prev_layer, filter = weights, strides = [1, 1, 1, 1], padding = 'SAME')
		out = tf.nn.relu(conv + bias)
		setattr(self, layer_name, out)


	def avgpool(self, prev_layer, layer_name):
		with tf.variable_scope(layer_name) as scope:
			out = tf.nn.avg_pool(prev_layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
		setattr(self, layer_name, out)


	def load_graph(self):
		self.conv2d_relu(self.input_img, 0, 'conv1_1')
		self.conv2d_relu(self.conv1_1, 2, 'conv1_2')
		self.avgpool(self.conv1_2, 'avgpool1')
		self.conv2d_relu(self.avgpool1, 5, 'conv2_1')
		self.conv2d_relu(self.conv2_1, 7, 'conv2_2')
		self.avgpool(self.conv2_2, 'avgpool2')
		self.conv2d_relu(self.avgpool2, 10, 'conv3_1')
		self.conv2d_relu(self.conv3_1, 12, 'conv3_2')
		self.conv2d_relu(self.conv3_2, 14, 'conv3_3')
		self.conv2d_relu(self.conv3_3, 16, 'conv3_4')
		self.avgpool(self.conv3_4, 'avgpool3')
		self.conv2d_relu(self.avgpool3, 19, 'conv4_1')
		self.conv2d_relu(self.conv4_1, 21, 'conv4_2')
		self.conv2d_relu(self.conv4_2, 23, 'conv4_3')
		self.conv2d_relu(self.conv4_3, 25, 'conv4_4')
		self.avgpool(self.conv4_4, 'avgpool4')
		self.conv2d_relu(self.avgpool4, 28, 'conv5_1')
		self.conv2d_relu(self.conv5_1, 30, 'conv5_2')
		self.conv2d_relu(self.conv5_2, 32, 'conv5_3')
		self.conv2d_relu(self.conv5_3, 34, 'conv5_4')
		self.avgpool(self.conv5_4, 'avgpool5')


