#!python
#encoding:utf-8

import os
import sys
import tensorflow as tf 
import argparse



#initilize the graph and model
class Model(object):
	def __init__(self,session,args):
		self.sess = session
		self.args = args
		self.initialize_graph()


	def conv(self,filter,feature_in,is_train,layer):
		with name_scope(layer):
			assert feature_in.get_shape()[3] == filter[2]," the 3 dim must be equal 2 dim "
			weights = tf.get_variable(name = "weights" , shape = filter )
			bias = tf.get_variable(name = "bias" , shape = filter[3] )
			cout = tf.nn.conv2d(feature_in , w , strides=[1,2,2,1] , padding = "VALID")
			result = tf.nn.bias_add(cout ,bias)

			result = tf.nn.relu(result)

		return tf.contrib.layers.batch_norm(result,center=True,decay=0.999,is_training=is_train,update_collections = None)



	def initialize_graph(self):

		input_x = tf.placeholder(shape=[None,args.width,args,height,args.channel],name="input")
		input_y = tf.placeholder(shape = [None , 1 ],name="label")
		input_box = tf.placeholder(shape = [None,1,1,1,1],name="location")


		if self.args.is_train:
			flag = True
			layer1 = conv([3,3,3,64],input_x,flag,"layer1")
			layer2 = conv([3,3,64,64],input_x,flag,"layer2")
			layer3 = conv([3,3,64,128],input_x,flag,"layer3")
			layer4 = conv([3,3,128,128],input_x,flag,"layer4")
			layer5 = conv([3,3,128,256],input_x,flag,"layer5")
			layer6 = conv([3,3,256,256],input_x,flag,"layer6")
			layer7 = conv([3,3,256,512],input_x,flag,"layer7")
			layer8 = conv([3,3,512,512],input_x,flag,"layer8")
			layer9 = conv([3,3,512,1024],input_x,flag,"layer9")
			layer10 = conv([3,3,1024,1024],input_x,flag,"layer10")






if__name__ == "__main__":
	parser = argparse.ArgumentParser("define all hyperparameters maybe used")
	parser.add_argument("-train",Type = bool, help="True if you want to train a model ")

	args = parser.parse_args()

	if args.train:
		with tf.Session() as sess:
			model = Model(sess,args)
