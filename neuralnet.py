from __future__ import division
import arff
import math
import copy
import sys

def sigmoid(x):
    return 1.0/(1.0+math.e**(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

class NeuralNetwork:

    def __init__(self, raw_data, n, l, e):

        self.__raw_data = None
        self.__data = list()
        self.__attribute_dictionary = dict()
        self.data_activation = []
        self.index = 0

        self.n = n
        self.l = l
        self.e = e
        self.pindex = 0
        self.num_pos = 0.0
        self.num_neg = 0.0

        self.train_accuracy_list = []
        self.test_accuracy_list = []

        self.__raw_data = copy.deepcopy(raw_data)

        # Get data in a cleaner format
        for d in self.__raw_data['data']:
            subject = {}
            for index, attribute in enumerate(self.__raw_data['attributes']):
                subject[attribute[0]] = d[index]
                self.__attribute_dictionary[attribute[0]] = attribute
            self.__data.append(subject)

        self.predictedclass = [None]*len(self.__data)

        #stratify
        folds, foldinfo = self.prepdata(n)
        testdataroc = []
        for fid in range(1,n+1):
            #initialize all the weights as 0.1
            self.weights = [0.1 for attr in self.__raw_data['attributes']]
            self.bias = 0.1
            train_data , test_data = self.makefolds(folds, n, fid)

            testdataroc = self.ann(train_data, test_data) + testdataroc

        #print "final train " + str((sum(self.train_accuracy_list)/len(self.train_accuracy_list)))
        #print "final test" + str((sum(self.test_accuracy_list)/len(self.test_accuracy_list)))
        self.stats(testdataroc)
        self.data_activation.sort(key=lambda tup: tup[1], reverse=True)

        #self.stats(train_data_list)
        self.roc(self.num_pos, self.num_neg)
        #self.test()
        #print the required information
        dataprintidx =0
        acc = 0
        for dataprint in self.__data:
            act = self.annactivation(dataprint)
            if act > 0.5:
                pclass = 1
            else:
                pclass = 0
            if pclass == dataprint["Class"]:
                acc+=1
            print str(foldinfo[dataprintidx]) + " " + str(pclass) + " "+ str(dataprint["Class"]) + " "+ str(act)
            dataprintidx+=1
        #print acc/len(self.__data)

    def prepdata(self, n):

        neg_data = [index for index, row in enumerate(self.__data) if row["Class"] == 0]
        pos_data = [index for index, row in enumerate(self.__data) if row["Class"] == 1]

        foldnegindex = []
        while (len(foldnegindex)< len(neg_data)):
            for i in range(1,n+1):
                foldnegindex.append(i)

        foldposindex = []
        while (len(foldposindex)< len(pos_data)):
            for i in range(1,n+1):
                foldposindex.append(i)

        neg_data_fold ={}
        pos_data_fold ={}

        for i in range(1,n+1):
            neg_data_fold_list =[]
            for lenneg in range(len(neg_data)):
                if foldnegindex[lenneg] == i:
                    neg_data_fold_list.append(neg_data[lenneg])
                    neg_data_fold[i] = neg_data_fold_list

        for j in range(1,n+1):
            pos_data_fold_list =[]
            for lenpos in range(len(pos_data)):
                if foldposindex[lenpos] == j:
                    pos_data_fold_list.append(pos_data[lenpos])
                    pos_data_fold[j] = pos_data_fold_list

        #match up the pos and neg from each folds
        folds = {}
        for key in range(1,n+1):
            folds[key] = neg_data_fold[key] + pos_data_fold[key]

        foldinfo = [None] * len(self.__data)
        for idx in folds:
            listf = folds[idx]
            for ilx in range(len(listf)):
                foldinfo[listf[ilx]] = idx

        return (folds, foldinfo)

    def makefolds(self, folds, n, fid):
        train_data = []
        #take each fold and then make it a test and everything else as training data
        test_data = folds[fid]

        for restid in range(1,n+1):
            if restid != fid:
                train_data = folds[restid] + train_data

        return (train_data, test_data)

    def annactivation(self, a):
        sum = 0
        i=0
        for key in a:
            sum = float(a[key] * self.weights[i]) + sum
            i= i+1

        activation = float(sigmoid(sum +self.bias))
        return activation

    def test(self):
        # repeat for epoch number of times
        for i in range(self.e):
            for data in self.__data:
                activation = self.annactivation(data)
                wt = 0
                #update the weights for the next instance
                for attr in data:
                    self.weights[wt] = float((self.l)*(data["Class"] - activation)*(activation)*(1-activation)*(data[attr])) + self.weights[wt]
                    wt = wt+1
                self.bias = float((self.l)*(data["Class"] - activation)*activation*(1-activation)) + self.bias
        final = self.predict(self.__data, 1)
        print "actual" + str(final)

    def ann(self, traindataindex, testdataindex):

        #we have the train data that we use to update the weights
        traindata = []
        for index in traindataindex:
            traindata.append(self.__data[index])

        # repeat for epoch number of times
        for i in range(self.e):
            for data in traindata:
                activation = self.annactivation(data)
                wt = 0
                #update the weights for the next instance
                for attr in data:
                    self.weights[wt] = float((self.l)*(data["Class"] - activation)*(activation)*(1-activation)*(data[attr])) + self.weights[wt]
                    wt = wt+1
                self.bias = float((self.l)*(data["Class"] - activation)*activation*(1-activation)) + self.bias

        train_accuracy = self.predict(traindata, 0)
        self.train_accuracy_list.append(train_accuracy)

        print "train " + str (train_accuracy)

        testdata = []
        for index in testdataindex:
            testdata.append(self.__data[index])

        test_accuracy = self.predict(testdata, 1)
        self.test_accuracy_list.append(test_accuracy)
        print test_accuracy
        return testdata

    def predict(self, testdata, f):

        #calculate the output for each instance
        num_accurate = 0
        total_instance =0

        for data in testdata:

            activation = self.annactivation(data)
            if activation > 0.5:
                if f ==1:
                    self.predictedclass[self.pindex] = 1
                if data["Class"] == 1:
                    num_accurate += 1
            else:
                if f==1:
                    self.predictedclass[self.pindex] = 0
                if data["Class"] == 0:
                    num_accurate += 1
            total_instance += 1
            if f ==1:
                activationtup = (self.index, activation)
                self.index = self.index + 1
            if f ==1:
                self.pindex += 1
                self.data_activation.append(activationtup)
        return float(num_accurate)/total_instance

    def stats(self, testdata):
        fn = 0
        fp = 0
        tp = 0
        tn = 0
        ifx = 0
        sorttestdata = []
        for index in self.data_activation:
            sorttestdata.append(testdata[index[0]])

        for sortdata in sorttestdata:

            if sortdata["Class"] == 1:
                self.num_pos += 1
            else:
                self.num_neg += 1

            if self.data_activation[ifx] > 0.5:
                if sortdata["Class"] == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if sortdata["Class"] == 0:
                    tn = tn + 1
                else:
                    fn = fn + 1
            ifx += 1

    # ROC calculation
    def roc(self, num_pos, num_neg):
        sorttestdata = [self.__data[d[0]] for d in self.data_activation]
        #make sorttestdata from indicies in data_activation

        FPR = 0.0
        TPR = 0.0
        roc_tp = 0.0
        roc_fp = 0.0
        last_tp = 0.0
        idgx = 0
        for rocsortdata in sorttestdata:
            if ((idgx >0) and (self.data_activation[idgx] != self.data_activation[idgx-1]) and (rocsortdata["Class"] == 0) and (roc_tp > last_tp)):
                FPR = float(roc_fp)/num_neg
                TPR = float(roc_tp)/num_pos
                #print str(FPR) +"," + str(TPR)
                last_tp = roc_tp
            if rocsortdata["Class"] == 1:
                roc_tp += 1
            else:
                roc_fp += 1
            idgx += 1
        FPR = float(roc_fp)/(num_neg)
        TPR = float(roc_tp)/(num_pos)
        #print str(FPR) +"," + str(TPR)


if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        raw_data = arff.load(f, 'rb')

    #fp = open('sonar.arff', 'rb')
    #raw_data = arff.load(fp, 'rb')
    #fp.close()
    # number of folds for cross validation
    n = (sys.argv[2])

    # learning rate
    l = (sys.argv[3])

    # number of training epochs
    e = (sys.argv[4])

    nn = NeuralNetwork(raw_data, 10, 0.1, 1)
