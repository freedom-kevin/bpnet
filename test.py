from __future__ import division
import train
import time

if __name__ == '__main__':
    input = 28*28
    output = 10
    testnet = train.NeuralNet([input, 40,output])
    test_set = train.load_samples(dataset='testing_data')
    testnet.load()

    correct = 0;
    begin_time=time.time()
    for test_feature in test_set:
        if testnet.predict(test_feature[0]) == test_feature[1]:
            correct += 1
    print("Error for test data is: {0}".format(1-correct/len(test_set)))
    print("Test takes: {0}s".format(time.time()-begin_time))
