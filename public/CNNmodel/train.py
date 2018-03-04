import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import vggnet
import utils
import time



class StyleTransfer(object):
	def __init__(self, content_img, style_img, img_height, img_width):

		# init imgs
		self.img_width = img_width
		self.img_height = img_height
		self.content_img = utils.resize_image(content_img, img_height, img_width)
		self.style_img = utils.resize_image(style_img, img_height, img_width)
		self.init_img = utils.blur_image(self.content_img, img_height, img_width)

		# init parameters
		self.content_layer = 'conv2_1'
		self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
		self.content_w = 0.01
		self.style_w = 1
		self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]
		self.gstep = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')
		self.lr = 2


	def generate_input(self):
		with tf.variable_scope('input') as scope:
			self.input_img = tf.get_variable('input_img',
											shape = [1, self.img_height, self.img_width, 3],
											dtype = tf.float32,
											initializer = tf.zeros_initializer())

	def load_vggnet(self):
		self.vgg = vggnet.VGGnet(self.input_img)
		self.vgg.load_graph()
		self.content_img -= self.vgg.mean_pixels
		self.style_img -= self.vgg.mean_pixels


	def _gram_matrix(self, F, N, M):
		F = tf.reshape(F, (M, N))
		return tf.matmul(tf.transpose(F), F)


	def _per_style_loss(self, s, g):
		N = s.shape[3]
		M = s.shape[1] * s.shape[2]
		gram_mat_s = self._gram_matrix(s, N, M)
		gram_mat_g = self._gram_matrix(g, N, M)
		return tf.reduce_sum((gram_mat_s - gram_mat_g) ** 2) / (4 * N**2 * M**2)


	def _tot_style_loss(self, S, G):
		n = len(self.style_layers)
		E = [self._per_style_loss(S[i], G[i]) for i in range(n)]
		self.style_loss = sum(self.style_layer_w[i] * E[i] for i in range(n))


	def _tot_content_loss(self, C, G):
		self.content_loss = (1/ 4 / C.size) * tf.reduce_sum(tf.square(C - G))


	def losses(self):
		with tf.variable_scope('losses') as scope:
			with tf.Session() as sess:	
				sess.run(self.input_img.assign(self.content_img))
				gen_img_content = getattr(self.vgg, self.content_layer)
				content_img_content = sess.run(getattr(self.vgg, self.content_layer))
			self._tot_content_loss(content_img_content, gen_img_content)

			with tf.Session() as sess:
				sess.run(self.input_img.assign(self.style_img))
				gen_img_style = [getattr(self.vgg, layer) for layer in self.style_layers]
				style_img_style = sess.run(gen_img_style)
			self._tot_style_loss(style_img_style, gen_img_style)

			self.total_loss = self.content_w * self.content_loss + self.style_w * self.style_loss


	def optimize(self):
		self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, global_step = self.gstep)


	def create_summary(self):
		with tf.name_scope('summaries'):
			tf.summary.scalar('content loss', self.content_loss)
			tf.summary.scalar('style loss', self.style_loss)
			tf.summary.scalar('total loss', self.total_loss)
			self.summary_op = tf.summary.merge_all()

	def buildNN(self):
		self.generate_input()
		self.load_vggnet()
		self.losses()
		self.optimize()
		self.create_summary()


	def train(self, iters):
		save_step = 20
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			writer = tf.summary.FileWriter('graphs', sess.graph)
			sess.run(self.input_img.assign(self.init_img))

			start_time = time.time()
			for iter in range(iters):
				sess.run(self.opt)
				if (iter + 1) % save_step == 0:
					total_loss, summary = sess.run([self.total_loss, self.summary_op])
					writer.add_summary(summary, global_step = iter)
					print('Step: {}  Loss: {:5.1f}'.format(iter + 1, total_loss))
					print('   Took: {} seconds'.format(time.time() - start_time))
					start_time = time.time()
			gen_image = sess.run(self.input_img)
			filename = '../public/output/generated_img.png'
			utils.save_image(filename, gen_image) 	


def main():
	model = StyleTransfer('../public/input/input.jpg', 'styles/starry_night.jpg', 250, 333)
	model.buildNN()
	model.train(20)



if __name__ == '__main__':
	main()