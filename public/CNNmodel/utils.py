import os
import numpy as np
import scipy.misc
from PIL import Image, ImageOps


def resize_image(img_path, height, width):
	image = Image.open(img_path)
	image = ImageOps.fit(image, (width, height), method = Image.ANTIALIAS)
	image_save_dir = img_path.split('/')
	image_save_dir[-1] = 'resized_' + image_save_dir[-1]
	output_path = "/".join(image_save_dir)
	if not os.path.exists(output_path):
		image.save(output_path)
	image = np.asarray(image, np.float32)
	# make the input image a volume of size [1, height, width]
	return np.expand_dims(image, 0)						

def blur_image(content_image, height, width, noise_ratio = 0.6):
	noise_image = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)
	return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def save_image(path, image):
	image = image[0]
	image = np.clip(image, 0, 255).astype('uint8')
	scipy.misc.imsave(path, image)

