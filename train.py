from __future__ import division
import numpy as np
import random
import os, struct,pickle
from array import array as pyarray
 
class NeuralNet(object):
 

    def __init__(self, sizes):
        self.sizes_ = sizes
        self.num_layers_ = len(sizes)  #the number of piles
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  #normal distribution
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]
 
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
 
    def feedforward(self, x):
        for b, w in zip(self.b_, self.w_):
            x = self.sigmoid(np.dot(w, x)+b)
        return x
 
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]
 
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.b_, self.w_):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
 
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
 
        for l in range(2, self.num_layers_):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.w_[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
 
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.w_ = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.w_, nabla_w)]
        self.b_ = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.b_, nabla_b)]

    #eta->learning rate
    def SGD(self, training_data, epochs, mini_batch_size,test_interval,eta, test_data=None):
        if test_data:
            n_test = len(test_data)
 
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if j%test_interval==0:
                print("Error for train data after Epoch {0} is: {1}".format(j, 1-self.evaluate(test_data)/n_test))

            else:
                print("Epoch {0} complete".format(j))
        self.save()
 
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
 
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
 
    # predict
    def predict(self, data):
        value = self.feedforward(data)
        return value.tolist().index(max(value))

 # save the model
    def save(self):
        modelfile=open('XuNet.model','wb')
        pickle.dump(self.w_,modelfile,False)
        pickle.dump(self.b_,modelfile,False)
        modelfile.close()
    # load the model
    def load(self):
        modelfile=open('XuNet.model','rb')
        self.w_=pickle.load(modelfile)
        self.b_=pickle.load(modelfile)
        modelfile.close()
 
def load_mnist(dataset="training_data", digits=np.arange(10), path=""):
 
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images.idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_label = os.path.join(path, 't10k-labels.idx1-ubyte')
    elif dataset == "validating_data":
        fname_image = os.path.join(path, 'train-images.idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")
 
    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
 
    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    #discard invalid data
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
 
    images = np.zeros((N, rows, cols), dtype='uint8')
    labels = np.zeros((N, 1), dtype='int8')
    for i in range(N):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
 
    return images, labels
 
def load_samples(dataset="training_data"):
 
    image,label = load_mnist(dataset)
 
    X = [np.reshape(x,(28*28, 1)) for x in image]
    X = [x/255.0 for x in X]  
 
    # 5 -> [0,0,0,0,0,1.0,0,0,0]; 
    def vectorized_Y(y): 
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e
 
    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    elif dataset=='validating_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')
 
 
if __name__ == '__main__':

    input = 28*28
    output = 10
    net = NeuralNet([input, 40,output])
    train_set = load_samples(dataset='training_data')
    validation_set=load_samples(dataset='validating_data')
    epochs=70
    mini_batch_size=100
    test_interval=2
    net.SGD(train_set, epochs, mini_batch_size, test_interval,3.0,validation_set)

    #97%