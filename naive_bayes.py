from __future__ import division
import arff
import copy
import math
import operator
import sys

class NaiveBayes:

    def __init__(self, raw_data, test_data):

        self.__raw_data = None
        self.__data = list()
        self.__attribute_dictionary = dict()

        self.__test_data = None
        self.__testdata = list()


        self.__raw_data = copy.deepcopy(raw_data)
        self.__test_data = copy.deepcopy(test_data)

        # Get data in a cleaner format
        for d in self.__raw_data['data']:
            subject = {}
            for index, attribute in enumerate(self.__raw_data['attributes']):
                subject[attribute[0]] = d[index]
                self.__attribute_dictionary[attribute[0]] = attribute
            self.__data.append(subject)

        for attribute in self.__raw_data['attributes']:
            if attribute[0] == "class":
                continue
            print str(attribute[0]), " class"

        for d in self.__test_data['data']:
            testsubject = {}
            for index, attribute in enumerate(self.__test_data['attributes']):
                testsubject[attribute[0]] = d[index]
            self.__testdata.append(testsubject)

        self.findProb()

    #find probability for each attribute's value and for each class
    def findProb(self):

        probClass = {}
        numClass = {}

        for val in range(len(self.__attribute_dictionary['class'][1])):
            probClass[val] =0
        total= 0
        #find the class probability
        for val in probClass:
            for data in self.__data:

                if data['class'] == val:
                    probClass[val] += 1
            total = probClass[val] +total
        #calculating the laplace probability for class
        for val in probClass:
            numClass[val] = (probClass[val]+1)/(total+2)
        #print numClass

        self.probAttr= dict()


        for attrname, attr in self.__attribute_dictionary.iteritems():
            if attrname == 'class':
                continue

            self.probAttr[attrname] = dict()
            for i in range(len(attr[1])):
                #go through the entire data

                self.probAttr[attrname][attr[1][i]] = list()
                num_pos=0
                num_neg=0
                for data in self.__data:
                    if data[attrname]==i:
                        if data['class'] == 1:
                            num_pos +=1
                        else:
                            num_neg +=1
                self.probAttr[attrname][attr[1][i]].append(num_neg)
                self.probAttr[attrname][attr[1][i]].append(num_pos)

                #print attrname, attr[1][i], i, num_pos, num_neg
        #print attrname

        self.LapProb = {}

        for attrname, attr in self.probAttr.iteritems():
            self.LapProb[attrname] = dict()

            for key, val in attr.iteritems():
                #go through all the attribute values
                self.LapProb[attrname][key] = list()
                for clsIdx in range(2):
                    #print val[clsIdx],probClass[clsIdx], len(attr),(val[clsIdx] + 1)/(probClass[clsIdx] + len(attr))
                    self.LapProb[attrname][key].append((val[clsIdx] + 1)/(probClass[clsIdx] + len(attr)))

        #classify
        predRight=0
        for d in self.__testdata:

            prob = numClass[d["class"]]
            for val in d:
                if val == "class":
                    continue
                for k,v in enumerate(self.__attribute_dictionary[val][1]):
                    if k == d[val]:
                        prob  = self.LapProb[val][v][d["class"]] * prob
                        #print val, d[val], v
                        #print self.LapProb[val][v][d["class"]]
                        break


            #print prob, d["class"]

            if d["class"]==0:
                inv_prob = 1
            else:
                inv_prob = 0
            prob2 = numClass[inv_prob]

            for val in d:
                if val == "class":
                    continue
                for k,v in enumerate(self.__attribute_dictionary[val][1]):
                    if k == d[val]:
                        prob2  = self.LapProb[val][v][inv_prob] * prob2
                        #print val, v
                        #print self.LapProb[val][v][d["class"]]
                        break
            #print prob2
            #print "prob:" + str(prob)
            #print "prob2: " + str(prob2)
            #print "prob: " + str(prob/prob2)

            if prob/(prob+prob2) > 0.5:
                predclass= d["class"]
                predprob = prob/(prob+prob2)
                predRight +=1
            else:
                predclass=abs(1-d["class"])
                predprob = prob2/(prob+prob2)


            print str(self.__attribute_dictionary["class"][1][predclass])+ " " + \
                  str(self.__attribute_dictionary["class"][1][d["class"]]) + " " + str("%.16f" %predprob)

        print predRight


class Tan():
    def __init__(self, raw_data, test_data):

        self.__raw_data = None
        self.__data = list()
        self.__attribute_dictionary = dict()

        self.__test_data = None
        self.__testdata = list()


        self.__raw_data = copy.deepcopy(raw_data)
        self.__test_data = copy.deepcopy(test_data)

        # Get data in a cleaner format
        for d in self.__raw_data['data']:
            subject = {}
            for index, attribute in enumerate(self.__raw_data['attributes']):
                subject[attribute[0]] = d[index]
                self.__attribute_dictionary[attribute[0]] = attribute
            self.__data.append(subject)

        for d in self.__test_data['data']:
            testsubject = {}
            for index, attribute in enumerate(self.__test_data['attributes']):
                testsubject[attribute[0]] = d[index]
            self.__testdata.append(testsubject)

        self.findProb()


    #find probability for each attribute's value and for each class
    def findProb(self):

        probClass = {}
        numClass = {}

        for val in range(len(self.__attribute_dictionary['class'][1])):
            probClass[val] =0
        total= 0
        #find the class probability
        for val in probClass:
            for data in self.__data:

                if data['class'] == val:
                    probClass[val] += 1
            total = probClass[val] +total
        #calculating the laplace probability for class
        for val in probClass:
            numClass[val] = (probClass[val]+1)/(total+2)
        #print numClass

        self.probAttr= dict()


        for attrname, attr in self.__attribute_dictionary.iteritems():
            if attrname == 'class':
                continue

            self.probAttr[attrname] = dict()
            for i in range(len(attr[1])):
                #go through the entire data

                self.probAttr[attrname][attr[1][i]] = list()
                num_pos=0
                num_neg=0
                for data in self.__data:
                    if data[attrname]==i:
                        if data['class'] == 1:
                            num_pos +=1
                        else:
                            num_neg +=1
                self.probAttr[attrname][attr[1][i]].append(num_neg)
                self.probAttr[attrname][attr[1][i]].append(num_pos)

                #print attrname, attr[1][i], i, num_pos, num_neg
        #print attrname

        self.LapProb = {}

        for attrname, attr in self.probAttr.iteritems():
            self.LapProb[attrname] = dict()

            for key, val in attr.iteritems():
                #go through all the attribute values
                self.LapProb[attrname][key] = list()
                for clsIdx in range(2):
                    #print val[clsIdx],probClass[clsIdx], len(attr),(val[clsIdx] + 1)/(probClass[clsIdx] + len(attr))
                    self.LapProb[attrname][key].append((val[clsIdx] + 1)/(probClass[clsIdx] + len(attr)))

        #calculating conditional mutual information
        self.TanProb = {}

        for attrname1, attr1 in self.__attribute_dictionary.iteritems():
            self.TanProb[attrname1] = {}
            for attrname2, attr2 in self.__attribute_dictionary.iteritems():
                if attrname1 == 'class' or attrname2 == 'class':
                   continue


                self.TanProb[attrname1][attrname2] = []
                mutualInfo = 0
                for i in range(len(attr1[1])):
                    for j in range(len(attr2[1])):
                        #go through the entire data
                        num_pos=0
                        num_neg=0
                        totalInst =0
                        for data in self.__data:
                            totalInst +=1
                            if data[attrname1]==i and data[attrname2]==j:
                                if data['class'] == 1:
                                    num_pos +=1
                                else:
                                    num_neg +=1
                        #calculate the P(x1,x2|y)
                        mutProb =[]
                        eachattr1 = []
                        eachattr2 = []
                        probAttr = []
                        mutProb.append((num_neg + 1)/((probClass[0])+(len(attr1[1])*len(attr2[1]))))
                        mutProb.append((num_pos + 1)/((probClass[1])+(len(attr1[1])*len(attr2[1]))))
                        eachattr1.append(self.LapProb[attrname1][attr1[1][i]][0])
                        eachattr1.append(self.LapProb[attrname1][attr1[1][i]][1])
                        eachattr2.append(self.LapProb[attrname2][attr2[1][j]][0])
                        eachattr2.append(self.LapProb[attrname2][attr2[1][j]][1])
                        probAttr.append((num_neg+1)/(totalInst+(len(probClass)*len(attr1[1])*len(attr2[1]))))
                        probAttr.append((num_pos+1)/(totalInst+(len(probClass)*len(attr1[1])*len(attr2[1]))))

                        mutualInfo += probAttr[0]*(math.log((mutProb[0]/(eachattr1[0]*eachattr2[0])),2)) + \
                                    probAttr[1]*(math.log((mutProb[1]/(eachattr1[1]*eachattr2[1])),2))

                if attrname1==attrname2:
                    self.TanProb[attrname1][attrname2].append(-1)
                else:
                    self.TanProb[attrname1][attrname2].append(mutualInfo)
        #making adjacency matrix
        temp = [[]]
        for attr1 in self.__raw_data["attributes"]:
            if attr1[0] == "class":
                continue
            ttemp=[]
            for attr2 in self.__raw_data["attributes"]:
                if attr2[0] == "class":
                    continue
                ttemp.append(self.TanProb[attr1[0]][attr2[0]][0])
            temp.append(ttemp)
        temp.remove([])
        #print temp

        Vertices = []
        Vertices.append(0)
        Tree={}
        Edges = []
        maxVal=-1
        maxattr = self.__raw_data["attributes"][0][0]

        while(len(Vertices)<len(self.__raw_data["attributes"])-1):
            potential_edges = []
            for source in Vertices:
                for col in temp[source]:
                    if temp[source].index(col) not in Vertices:
                        potential_edges.append([source, temp[source].index(col), temp[source][temp[source].index(col)]])
            potential_edges.sort(key = lambda x:x[2])
            if (len(potential_edges) == 0):
                continue
            Vertices.append(potential_edges[-1][1])
            Edges.append([potential_edges[-1][0],potential_edges[-1][1]])
            #print Edges
        #print "test"


        SortedEdges = copy.deepcopy(Edges)
        SortedEdges.sort(key= lambda x:x[1])
        self.TanParent = []
        for i in range(len(self.__raw_data["attributes"])-1):
            found = None
            for e in SortedEdges:
                if e[1] == i:
                    print (self.__raw_data["attributes"][i][0] ) + " " +(self.__raw_data["attributes"][e[0]][0]) + " class"
                    self.TanParent.append([self.__raw_data["attributes"][i][0] , self.__raw_data["attributes"][e[0]][0], "class"])
                    found = True
                    break
            if not found:
                print (self.__raw_data["attributes"][i][0] ) + " class"
                self.TanParent.append([self.__raw_data["attributes"][i][0], "class"])

        #classify
        correctPred=0
        for dt in self.__testdata:
            mult1=1
            mult0=1
            for testattr in dt:
                if testattr == "class":
                    continue
                parVal=[]
                par=[]
                ParentNum = 0
                VarParentNum = 0
                invParentNum = 0
                invVarParentNum = 0

                for parent in self.TanParent:
                    if testattr == parent[0]:
                        for i in range(len(parent)-1):
                            var = dt[testattr]
                            par.append(parent[i+1])
                            parVal.append(dt[parent[i+1]])

                        for data in self.__data:
                            if data[testattr]!=var:
                                notset2 = False
                            else:
                                notset2 = True

                            for i in range(len(parent)-1):
                                if data[par[i]]!=parVal[i]:
                                    notset1 = False
                                    break
                                else:
                                    notset1 = True

                            if notset1 == True:
                                ParentNum +=1
                            if notset1 == True and notset2 == True:
                                VarParentNum +=1

                            #calculate for inverted class
                            parValCopy = copy.deepcopy(parVal)
                            for i in range(len(parent)-1):
                                if par[i] == "class":
                                    parValCopy[i]=abs(1-parVal[i])
                                if data[par[i]]!=parValCopy[i]:
                                    invnotset1 = False
                                    break
                                else:
                                    invnotset1 = True
                            if invnotset1 == True:
                                invParentNum +=1
                            if invnotset1 == True and notset2 == True:
                                invVarParentNum +=1

                        #print VarParentNum, ParentNum, invParentNum, invVarParentNum, varNum

                        posterior1 = (VarParentNum+1)/(ParentNum+len(self.__attribute_dictionary[testattr][1]))
                        posterior2 = (invVarParentNum+1)/(invParentNum+len(self.__attribute_dictionary[testattr][1]))
                        #print testattr, dt[testattr], parent[1], dt[parent[1]]
                        #print str("%.16f" % posterior1)
                        #print str("%.16f" % posterior2)
                        mult0 = mult0 * posterior1
                        mult1 = mult1 * posterior2

            #print numClass[dt["class"]],numClass[abs(1-dt["class"])], numClass[dt["class"]] * mult0, numClass[abs(1-dt["class"])] * mult1
            deno = (numClass[dt["class"]] * mult0) + (numClass[abs(1-dt["class"])] * mult1)
            predTanProb = (numClass[dt["class"]] * mult0)/(deno)


            if predTanProb > 0.5:
                correctPred += 1
                print self.__attribute_dictionary["class"][1][dt["class"]] +" "+  \
                      self.__attribute_dictionary["class"][1][dt["class"]] +" "+  str("%.16f" % predTanProb)
            elif predTanProb == 0.5:
                correctPred += 1
                print self.__attribute_dictionary["class"][1][0] + " " + \
                    self.__attribute_dictionary["class"][1][dt["class"]] +" " + str("%.16f" % predTanProb)
            else:
                print self.__attribute_dictionary["class"][1][abs(1-dt["class"])] +" "+  \
                    self.__attribute_dictionary["class"][1][dt["class"]] +" "+  str("%.16f" % (1-predTanProb))

        print correctPred


if __name__ == '__main__':
    '''
    fp = open('lymph_train.arff', 'rb')
    raw_data = arff.load(fp, 'rb')
    fp.close()
    fp = open('lymph_test.arff', 'rb')
    test_data = arff.load(fp, 'rb')
    fp.close()
    '''
    # train arff
    with open(sys.argv[1], 'rb') as f:
        raw_data = arff.load(f, 'rb')


    # test arff
    with open(sys.argv[2], 'rb') as f:
        test_data = arff.load(f, 'rb')


    # naive b or tan
    l = (sys.argv[3])

    if l == 'n':
        nn = NaiveBayes(raw_data, test_data)
    if l == 't':
        nn = Tan(raw_data, test_data)
