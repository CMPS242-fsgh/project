#Naive Bayes smoothing. 
#To run: python classifier.py <datafile>

import numpy as np
import random
import sys, math
import os
import glob


class Classifier:
	def __init__(self, featureGenerator, alpha = 0.05):
		self.featureGenerator = featureGenerator
		self._C_SIZE = 0
		self._V_SIZE = 0
		self._classes_list = []
		self._classes_dict = {}
		self._vocab = {}
		self._is_smoothed = True
		self._alpha = alpha

	def setClasses(self, trainingData):
		for(label, line) in trainingData:
			if label not in self._classes_dict:
				self._classes_dict[label] = len(self._classes_list)
				self._classes_list.append(label)
		self._C_SIZE = len(self._classes_list)
		return
		
	def getClasses(self):
		return self._classes_list

	def setVocab(self, trainingData):
		index = 0
		for (label, line) in trainingData:
			line = self.featureGenerator.getFeatures(line)
			for item in line:
				#print(item)
				if(item not in self._vocab):
					self._vocab[item] = index
					index += 1
		self._V_SIZE = len(self._vocab)
		return

	def getVocab(self):
		return self._vocab

	def train(self, trainingData):
		pass

	def classify(self, testData, params):
		pass

	def getFeatures(self, data):
       		return self.featureGenerator.getFeatures(data)

	def smooth(self, numerator, denominator):
		if self._is_smoothed:
			return (numerator + self._alpha) / (denominator + (self._alpha * len(self._vocab)))
		else:
			return numerator / denominator
	
	def normalize(self, post_prob):
		post_prob = self.shift_log(post_prob)
		norm_constant = 0.0
		for i in range(0, len(post_prob)):
			#print("Prob in log space: "+str(post_prob[i]))
			post_prob[i] = math.exp(post_prob[i])
			#print("Antilog of Prob: "+str(post_prob[i]))
			norm_constant +=  post_prob[i]

		for i in range(0, len(post_prob)):
			if(math.fabs(post_prob[i]) != float("inf") and norm_constant != 0):
				post_prob[i] = post_prob[i]/norm_constant
		return post_prob

	def shift_log(self, post_prob):
		maxProb = self.getMax(post_prob)
		for i in range(0, len(post_prob)):
			if(math.fabs(post_prob[i]) == float("inf") and math.fabs(maxProb) == float("inf")):
				post_prob[i] == -float("inf")
			#print("MaxProb: "+str(maxProb))
			else:
				post_prob[i] = post_prob[i] - maxProb
			#print(str(post_prob[i]))
		return post_prob
	
	def log(self, num):
		if num == 0:
			return -float("inf")
		elif num > 0:
			#print(str(num))
			return float(math.log(num))

	def getMaxIndex(self, posteriorProbabilities):
		maxi = 0
		maxProb = posteriorProbabilities[maxi]
		for i in range(0, len(posteriorProbabilities)):
			if(posteriorProbabilities[i] >= maxProb):
				maxProb = posteriorProbabilities[i]
				maxi = i
		return maxi

	def getMax(self, posteriorProbabilities):
		maxProb = -float("inf")
		for i in range(0, len(posteriorProbabilities)):
			if(posteriorProbabilities[i] >= maxProb):
				maxProb = posteriorProbabilities[i]
		#print("Max: "+str(maxProb))
		return maxProb

	def getAccuracy(self, actual_labels, pred_labels):
		acc = 0.0
		if(len(actual_labels) == len(pred_labels)):
			for i in range(0, len(actual_labels)):
				if(pred_labels[i] == actual_labels[i]):
					acc = acc+1
			acc = acc/len(actual_labels)	
			#print("Accuracy of the smoothed classifier is: "+str(acc))	
			return acc
		else:
			return(-1)

	def getConfMatrix(self, actual_labels, pred_labels):
		confMatrix = np.zeros((self._C_SIZE, self._C_SIZE))	
		for i in range(0, len(actual_labels)):
			confMatrix[self._classes_list.index(pred_labels[i])][self._classes_list.index(actual_labels[i])] = confMatrix[self._classes_list.index(pred_labels[i])][self._classes_list.index(actual_labels[i])] + 1
		#print("The classes are: "+str(self._classes_list))
		#print("The confusion matrix is: \n"+str(confMatrix))
		return confMatrix

	def getPrecision(self, confMatrix):
		prec_vec = []
		for i in range(0, self._C_SIZE):
			temp = 0
			for j in range(0, self._C_SIZE):
				temp = temp + confMatrix[i][j]
			if(temp != 0):
				prec_vec.append(confMatrix[i][i]/temp)
			else:
				prec_vec.append(0.0)
		#print("The classes are: "+str(self._classes_list))
		#print("Precision: "+str(prec_vec))
		return prec_vec

	def getRecall(self, confMatrix):
		recall_vec = []
		for j in range(0, self._C_SIZE):
			temp = 0
			for i in range(0, self._C_SIZE):
				temp = temp + confMatrix[i][j]
			if(temp != 0):
				recall_vec.append(confMatrix[j][j]/temp)
			else:
				recall_vec.append(0.0)
		#print("The classes are: "+str(self._classes_list))
		#print("Recall: "+str(recall_vec))
		return recall_vec

	def getFMeasure(self, prec, recall):
		fscore_vec = []
		for i in range(0, self._C_SIZE):
			if((prec[i] + recall[i]) != 0):
				fscore_vec.append(2.0*prec[i]*recall[i]/(prec[i] + recall[i]))
			else:
				fscore_vec.append(0.0)
		#print("The classes are: "+str(self._classes_list))
		#print("Fscore: "+str(fscore_vec))
		return fscore_vec
		

class FeatureGenerator:
	def getFeatures(self, text):
		text = text.lower()
		return text.split()

class NaiveBayesClassifier(Classifier):
	def __init__(self, fg, alpha=0.5):
		Classifier.__init__(self, fg, alpha)
		self._classParams = []
		self._params = [[]]

	def getParameters(self):
		return (self._classParams, self._params)

	def train(self, trainingData):
		self.setClasses(trainingData)
		self.setVocab(trainingData)
		self.initParameters()

		for (cat, document) in trainingData:
			for feature in self.featureGenerator.getFeatures(document):
				self.countFeature(feature, self._classes_dict[cat])

	def countFeature(self, feature, class_index):
		counts = 1
		self._counts_in_class[class_index][self._vocab[feature]] = self._counts_in_class[class_index][self._vocab[feature]] + counts
		self._total_counts[class_index] = self._total_counts[class_index] + counts
		self._norm = self._norm + counts

	def classify(self, testData):
		post_prob = self.getPosteriorProbabilities(testData)
		#print("NB: Posterior probabilities" +str(post_prob))
		maxi = self.getMaxIndex(post_prob)
		#print("NB: Max Posterior probability " +str(post_prob[maxi]))
		return self._classes_list[maxi]

	def getPosteriorProbabilities(self, testData):
		post_prob = self.getPosteriorProbabilitiesWithoutNormalization(testData)
		post_prob = self.normalize(post_prob)
		return post_prob

	def getPosteriorProbabilitiesWithoutNormalization(self, testData):
		post_prob = np.zeros(self._C_SIZE)
		for i in range(0, self._C_SIZE):
			for feature in self.getFeatures(testData):
				post_prob[i] += self.getLogProbability(feature, i)
				#print ("NB: f_given_c ( " + feature + "|" + str(i)+" ) {"+str(self._alpha)+ "," + str(len(self._vocab))+"} = "+str(self.getLogProbability(feature, i)))
			post_prob[i] += self.getClassLogProbability(i)
		return post_prob

	def getFeatures(self, testData):
		return self.featureGenerator.getFeatures(testData)

	def initParameters(self):
		self._total_counts = np.zeros(self._C_SIZE)
		self._counts_in_class = np.zeros((self._C_SIZE, self._V_SIZE))
		self._norm = 0.0

	def getLogProbability(self, feature, class_index):
		return self.log(self.getProbability(feature, class_index))
	

	def getProbability(self, feature, class_index):
		#print("Word count in class"+str(self.getCount(feature, class_index))+"Words in class"+str(self._total_counts[class_index]))
		return self.smooth(self.getCount(feature, class_index),self._total_counts[class_index])

	def getCount(self, feature, class_index):
		if feature not in self._vocab:
			return 0
		else:
			return self._counts_in_class[class_index][self._vocab[feature]]

	def getClassLogProbability(self, class_index):
		#print("total counts, norm: "+str(self._total_counts[class_index])+", "+str(self._norm))
		#print("return value: "+str(self.log(self._total_counts[class_index]/self._norm)))
		return self.log(self._total_counts[class_index]/self._norm)
	

	def getClassProbability(self, class_index):
		return self._total_counts[class_index]/self._norm

	def printParameters(self):
		self.computeParameters()
		for i in range(0, self._C_SIZE):
			print("\n"+"class["+str(self._classes_list[i])+"]: "+str(self._classParams[i]))
			for item in self._vocab:
				j = self._vocab[item]
				print(str(item)+": "+str(self._params[i][j]))

	def showBestFeatures(self):
		self.computeParameters()
		for i in range(0, self._C_SIZE):
			print("\n"+"class["+str(self._classes_list[i])+"]: ")
			print("\nBest features: ")
			best_features = self._params[i].argsort()[::-1][:10]
			for j in best_features:
				print(str(self._vocab.keys()[self._vocab.values().index(j)])+": "+str(self._params[i][j]))

	def computeParameters(self):
		self._classParams = np.zeros(self._C_SIZE)
		self._params = np.zeros((self._C_SIZE, self._V_SIZE))

		for i in range(0, self._C_SIZE):
			if(self._norm != 0):
				self._classParams[i] = self._total_counts[i]/self._norm
				for j in range(0, self._V_SIZE):
					if(self._total_counts[i] != 0):
						self._params[i][j] = self._counts_in_class[i][j]/self._total_counts[i]
					else:
						self._params[i][j] = 0
			else:
				self._classParams[i] = 0.0


class DataLoader:
	def __init__(self):
		self._dataset = []
		self._D_SIZE = 0
		#self._trainSIZE = int(0.6*self._D_SIZE)
		#self._testSIZE = int(0.3*self._D_SIZE)
		#self._devSIZE = 1 - (self._trainSIZE + self._testSIZE)

	def readData(self, datasource):
		pass

	def storeData(self, trainFile, testFile):
		#print(os.getcwd())
		fp = open(trainFile, "w")
		for (cat, line) in self.getTrainingData():
			fp.write(cat+" "+line+"\n")
		fp.close()

		fp = open(testFile, "w")
		for (cat, line) in self.getTestData():
			fp.write(cat+" "+line+"\n")
		fp.close()

	def settrainSIZE(self, value = 70):
		self._trainSIZE = int(value*0.01*self._D_SIZE)

	def settestSIZE(self, value = 30):
		self._testSIZE = int(value*0.01*self._D_SIZE)

	def setdevSIZE(self):
		self._devSIZE = int(1 - (self._trainSIZE + self._testSIZE))
		
	def getDataSIZE(self):
		self._D_SIZE = len(self._dataset)
		return self._D_SIZE

	def gettrainSIZE(self):
		return self._trainSIZE

	def gettestSIZE(self):
		return self._testSIZE

	def getdevSIZE(self):
		return self._devSIZE
	
	def getTrainingData(self):
		return self._dataset[0:self._trainSIZE]

	def getTestData(self):
		return self._dataset[self._trainSIZE:(self._trainSIZE+self._testSIZE)]

	def getDevData(self):
		return self._dataset[0:self._devSIZE]
					

class DataLoaderFromFile(DataLoader):
	def __init__(self):
		DataLoader.__init__(self)

	def readData(self, filename):
		fp = open(filename, "r")
		for line in fp:
			if(line != "\n"):
				line = line.split()
				cat = line[0]
				sent = ""
				for word in range(1, len(line)):
					sent = sent+line[word]+" "
				sent = sent.strip()
				self._dataset.append([cat, str(sent)])
		fp.close()
		random.shuffle(self._dataset)	
		self._D_SIZE = len(self._dataset)

	def loadTrainingData(self, trainFile):
		trainingData = []
		fp = open(trainFile, "r")
		for line in fp:
			if(line != "\n"):
				line = line.split()
				cat = line[0]
				sent = ""
				for word in range(1, len(line)):
					sent = sent+line[word]+" "
				sent = sent.strip()
				trainingData.append([cat, str(sent)])
				self._dataset.append([cat, str(sent)])
		fp.close()
		self._trainSIZE = len(trainingData)
		return trainingData

	def loadTestData(self, testFile):
		testData = []
		fp = open(testFile, "r")
		for line in fp:
			if(line != "\n"):
				line = line.split()
				cat = line[0]
				sent = ""
				for word in range(1, len(line)):
					sent = sent+line[word]+" "
				sent = sent.strip()
				testData.append([cat, str(sent)])
				self._dataset.append([cat, str(sent)])
		fp.close()
		self._testSIZE = len(testData)
		return testData
		

class DataLoaderFromDir(DataLoader):
	def __init__(self):
		DataLoader.__init__(self)

	def readData(self, datasource):
		os.chdir(datasource)
		total_feature_size = 0.0
		for folder in glob.glob("*"):
			os.chdir(folder)
			cat = folder
			for fname in glob.glob("*.txt"):
				fp = open(fname, "r")
				text = fp.read()
				text = text.replace("\n", "")
				#text = text.split()
				#total_feature_size += len(text)
				self._dataset.append([cat, text])
				fp.close()
			os.chdir("..")
		os.chdir("..")
		os.chdir("..")
		
		#print("Avg featuresize = "+str(total_feature_size/i))
		random.shuffle(self._dataset)	
		self._D_SIZE = len(self._dataset)


def getFeatures(data):
	data_features = []
	for (cat, document) in data:
		featureset = {}
		document = document.lower()
		feature = document.split()
		for item in feature:
			featureset.update({item : document.count(item)})
		data_features.append([featureset, cat])
	return data_features

#============================================================================================
if __name__ == "__main__":
	
	filename = "sample_file.txt"
	if len(sys.argv) > 1:
		filename = sys.argv[1]

	if(os.path.isfile(filename)):
		data = DataLoaderFromFile()
	else:
		data = DataLoaderFromDir()
	
	data.readData(filename)
	results_file = open(filename+"_predictions", "w")

	data.settrainSIZE(70)
	#data.settrainSIZE(100)
	train_set = data.getTrainingData()

	#dev_set = data.getDevData()

	data.settestSIZE(30)
	test_set = data.getTestData()

	#data.storeData(filename+"_train", filename+"_test")

	
	'''trainFile = "data/imdb_fixed_train"
	testFile = "data/imdb_fixed_test"

	#trainFile = "data/uiuc_train_fine"
	#testFile = "data/uiuc_test_finee"
	
	#trainFile = "data/20newsgroups.train"
	#testFile = "data/20newsgroups.test"

	#trainFile = "data/20newsgroups_revised.train"
	#testFile = "data/20newsgroups_revised.test"

	#trainFile = "data/Reuters_Apte.train.fixed"
	#testFile = "data/Reuters_Apte.test.fixed"

	#trainFile = "data/webkb-train-stemmed.txt"
	#testFile = "data/webkb-test-stemmed.txt"

	#trainFile = "data/reuters_R8.train"
	#testFile = "data/reuters_R8.test"

	#trainFile = "data/reuters_R52.train"
	#testFile = "data/reuters_R52.test"

	#trainFile = "data/Reuters_Apte.train"
	#testFile = "data/Reuters_Apte.test"
	
	data = DataLoaderFromFile()
	train_set = data.loadTrainingData(trainFile)
	test_set = data.loadTestData(testFile)
	'''

	test_data = [test_set[i][1] for i in range(len(test_set))]
	actual_labels = [test_set[i][0] for i in range(len(test_set))]

	fg = FeatureGenerator()
	
	alpha = 1.0 #smoothing parameter

	#data.storeData(filename+"_train", filename+"_test")

	nbClassifier = NaiveBayesClassifier(fg, alpha)

	'''nbClassifier.setClasses(train_set)
	nbClassifier.setVocab(train_set)
	print(nbClassifier.getVocab())
	print(str(len(nbClassifier.getVocab())))'''

	nbClassifier.train(train_set)
	#nbClassifier.showBestFeatures()
	
	pred_labels_nb = []
	i = 0
	for line in test_data:
		best_label = nbClassifier.classify(line)
		pred_labels_nb.append(best_label)
		#print(str(line)+" : "+str(best_label)+" (predicted)")
		#if(str(best_label)!= actual_labels[i]):
			#print(str(line)+" : "+str(best_label)+" (predicted)"+" "+actual_labels[i]+" (actual)")
		results_file.write(str(actual_labels[i])+"\t"+str(pred_labels_nb[i])+"\n")
		i=i+1
	acc = nbClassifier.getAccuracy(actual_labels, pred_labels_nb)
	print("The accuracy of the classifier is: "+str(acc))
	
	confMatrix = nbClassifier.getConfMatrix(actual_labels, pred_labels_nb)
	prec = nbClassifier.getPrecision(confMatrix)
	recall = nbClassifier.getRecall(confMatrix)
	#print(prec)
	#print(recall)
	#print(confMatrix)
	

	results_file.close()



