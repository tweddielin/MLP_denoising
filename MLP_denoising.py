from skimage import io
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from colorama import init
init(autoreset = True)
from colorama import Fore, Back, Style
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
import cPickle

class HiddenLayer(object):
	def __init__(self,rng, input, n_in, n_out, W = None, b = None, activation = T.tanh):
		self.input = input
		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low = -np.sqrt(6. / (n_in + n_out)),
					high = np.sqrt(6. / (n_in + n_out)),
					size = (n_in, n_out)
					),
				dtype = theano.config.floatX)

			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value = W_values, name = 'W', borrow = True)

		if b is None:
			b_values = np.zeros((n_out,), dtype = theano.config.floatX)
			b = theano.shared(value = b_values, name = 'b', borrow = True)

		self.W = W
		self.b = b
		#self.W2 = W2
		#self.b2 = b2
		lin_output = T.dot(input, self.W) + self.b
		#lin_output2 = T.dot(lin_output2, self.W2) + self.b2
		self.output = (activation(lin_output))
		self.params = [self.W, self.b]

class OutputLayer(object):
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(value=np.zeros((n_in, n_out),dtype=theano.config.floatX),name='W', borrow=True)
		self.b = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
		self.output = T.dot(input, self.W) + self.b
		self.params = [self.W, self.b]
		self.input = input
	def least_square(self, y):
		return T.mean(T.sqr(y - self.output))	

class MLP(object):
	def __init__(self, rng, input, n_in, n_hidden, n_out = 17*17):

		self.hiddenLayer = HiddenLayer(rng=rng, input =input, n_in = n_in, n_out = n_hidden, activation = T.tanh)
		self.outputLayer = OutputLayer(input=self.hiddenLayer.output,n_in=n_hidden,n_out=n_out)
		self.least_square = (self.outputLayer.least_square)
		self.output = (self.outputLayer.output)
		self.params = self.hiddenLayer.params + self.outputLayer.params
		self.L1 = (abs(self.hiddenLayer.W).sum() + abs(self.outputLayer.W).sum())
		self.L2_sqr = ((self.hiddenLayer.W ** 2).sum()+ (self.outputLayer.W ** 2).sum())
		self.input = input



def patchify(img, patch_shape,step):
	img = np.ascontiguousarray(img)
	X, Y =  img.shape
	x, y = patch_shape
	shape = ((X-x+1)/step, (Y-y+1)/step, x, y)
	#strides = img.itemsize * np.array([Y,1,Y,1])
	strides = (img.strides[0]*step, img.strides[1]*step, img.strides[0], img.strides[1])
	return np.lib.stride_tricks.as_strided(img, shape=shape, strides = strides)

def load_data(patchsize):
	input_image = io.imread('noiseboat.png')
	label_image = io.imread('boat.png')



	patches = patchify(input_image, (patchsize,patchsize),3)
	contiguous_patches = np.ascontiguousarray(patches)
	contiguous_patches.shape = (-1, patchsize**2)
	noisy_patches = contiguous_patches
	noisy_patches = noisy_patches/255.
	noisy_patches = (noisy_patches - 0.5)/0.2

	patches = patchify(label_image, (patchsize,patchsize),3)
	contiguous_patches = np.ascontiguousarray(patches)
	contiguous_patches.shape = (-1, patchsize**2)
	clean_patches = contiguous_patches
	clean_patches = clean_patches/255.
	clean_patches = (clean_patches - 0.5)/0.2

	train_noisy, test_noisy, train_clean, test_clean = train_test_split(noisy_patches, clean_patches, test_size = 0.33)
	print Style.BRIGHT + '# of training: ' + Fore.YELLOW + '%i' % len(train_noisy)
	print Style.BRIGHT + '# of testing: ' + Fore.YELLOW + '%i' % len(test_noisy)

	#tr0x = train_noisy[30].reshape(64,64)
	#tr0y = train_clean[30].reshape(64,64)
	#te0x = test_noisy[30].reshape(64,64)
	#te0y = test_clean[30].reshape(64,64)
	

	train_set = (train_noisy, train_clean)
	test_set = (test_noisy, test_clean)


	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
		return shared_x, T.cast(shared_y, 'float32')

	train_set_x, train_set_y = shared_dataset(train_set)
	test_set_x, test_set_y = shared_dataset(test_set)

	'''
	tr0x = train_set_x.get_value(borrow=True)[30].reshape(patchsize,patchsize)
	tr0y = train_set_y.get_value(borrow=True)[30].reshape(patchsize,patchsize)
	te0x = test_set_x.get_value(borrow=True)[30].reshape(patchsize,patchsize)
	te0y = test_set_y.get_value(borrow=True)[30].reshape(patchsize,patchsize)
	plt.figure()
	plt.imshow(np.hstack([tr0x,tr0y,te0x,te0y]),cmap = plt.get_cmap('gray'))
	plt.axis('off')
	plt.show()
	'''
	return [(train_set_x, train_set_y), (test_set_x, test_set_y)]

def test_mlp(learning_rate=0.01,L1_reg=0.00, L2_reg=0.0001, n_epochs=600, batch_size=80, n_hidden=511, patchsize = 17):
	datasets = load_data(patchsize)
	train_set_x, train_set_y = datasets[0]
	test_set_x, test_set_y = datasets[1]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	
	######################
    # BUILD ACTUAL MODEL #
    ######################
	print '... building the model'
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.matrix('y')

	rng = np.random.RandomState(1234)

	estimater = MLP(rng=rng,input=x,n_in=patchsize *  patchsize,n_hidden=n_hidden,n_out=patchsize * patchsize)
	cost = (estimater.least_square(y)+ L1_reg * estimater.L1 + L2_reg * estimater.L2_sqr)
	test_model = theano.function(inputs=[x],outputs=estimater.output)
	gparams = [T.grad(cost, param) for param in estimater.params]
	updates = [(param, param - learning_rate * gparam) for param, gparam in zip(estimater.params, gparams)]
	train_model = theano.function(inputs=[index],outputs=cost,updates=updates,givens={
    	x: train_set_x[index * batch_size: (index + 1) * batch_size],
    	y: train_set_y[index * batch_size: (index + 1) * batch_size]})

	#################
	# TRAINING MODE #
	#################
	print '... training'
	
	plt.figure()
	plt.ion()
	plt.axis('off')
	plt.show()
	k = 1
	for epoch in xrange(n_epochs):
		c = []
		
		for batch_index in xrange(n_train_batches):
			c.append(train_model(batch_index))
		
		n = (epoch*5 + k) % 5
		denoise_te = test_model([train_set_x.get_value(borrow=True)[n]]).reshape(patchsize,patchsize)
		clean_te = train_set_y.get_value(borrow=True)[n].reshape(patchsize,patchsize)
		noise_te = train_set_x.get_value(borrow=True)[n].reshape(patchsize,patchsize)
		plt.imshow(np.hstack([clean_te*255, noise_te*255, denoise_te*255]),cmap = plt.get_cmap('gray'))
		#plt.text(0.9,0.9,'epoch {0}'.format(epoch),fontsize = 20,color='k', backgroundcolor='w')
		plt.title('epoch {0}'.format(epoch), fontsize =20)
		plt.draw()

		if epoch % 5 == 0:
			k = k + 1 
		print (Style.BRIGHT + 'Training epoch ' + Fore.YELLOW + '%d,'+Fore.RESET+' cost ' + Fore.RED + '%f') % (epoch, np.mean(c))
	plt.close()	
	# Save the model
	with open('best_model.pkl', 'w') as f:
		cPickle.dump(test_model, f)


	for i in range(10):
		denoise_te = test_model([test_set_x.get_value(borrow=True)[i]]).reshape(patchsize,patchsize)
		clean_te = test_set_y.get_value(borrow=True)[i].reshape(patchsize,patchsize)
		noise_te = test_set_x.get_value(borrow=True)[i].reshape(patchsize,patchsize)
		denoise_tr = test_model([train_set_x.get_value(borrow=True)[i]]).reshape(patchsize,patchsize)
		clean_tr = train_set_y.get_value(borrow=True)[i].reshape(patchsize,patchsize)
		noise_tr = train_set_x.get_value(borrow=True)[i].reshape(patchsize,patchsize)

		fig = plt.figure()
		plt.ioff()
		axTe = fig.add_subplot(2,1,1)
		axTe.imshow(np.hstack([clean_te*255, noise_te*255, denoise_te*255]),cmap = plt.get_cmap('gray'))
		axTe.axis('off')
		axTe.text(0.08,0.9,'clean ', horizontalalignment='center', transform=axTe.transAxes, color ='b')
		axTe.text(0.42,0.9,'noise ', horizontalalignment='center', transform=axTe.transAxes, color ='b')
		axTe.text(0.76,0.9,'denoised ', horizontalalignment='center', transform=axTe.transAxes, color ='b')
		axTe.set_title('Some Testing Patch')
		

		axTr = fig.add_subplot(2,1,2)
		axTr.imshow(np.hstack([clean_tr*255, noise_tr*255, denoise_tr*255]),cmap = plt.get_cmap('gray'))
		axTr.axis('off')
		axTr.text(0.08,-0.38,'clean ', horizontalalignment='center', transform=axTe.transAxes, color ='b')
		axTr.text(0.42,-0.38,'noise ', horizontalalignment='center', transform=axTe.transAxes, color ='b')
		axTr.text(0.76,-0.38,'denoised ', horizontalalignment='center', transform=axTe.transAxes, color ='b')
		axTr.set_xlabel('clean noise denoised')
		axTr.set_title('Some Training Patch')
		
		
		plt.show()


if __name__ == '__main__':
    test_mlp()
    #load_data()





