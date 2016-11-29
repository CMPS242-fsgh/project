#!/usr/bin/env python2

import argparse
import sys
import time


import loader
from feature import CountFeature
from classifier import NaiveBayes
from multi import OneVsRest
from compute_Metrics import computeMetrics

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed

@timeit
def load_data(n, n_feature, per, vectorizer):
    YY, categories = loader.get_target(n_feature, n)
    XX = vectorizer(loader.review_from_file(n))
    X, Y, Xt, Yt = loader.split_training(XX, YY, per)
    return X, Y, Xt, Yt, categories

@timeit
def run_OneVsRest(data, ensembler, classifier):
    X, Y, Xt, Yt, cats = data
    model = ensembler(classifier)
    import warnings
    model.fit(X, Y)
    Yp = model.predict(Xt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hl = computeMetrics(Yp, Yt, cats)

    print 'the hamming loss:'
    print '>>  ', hl
    print 'DONE..'

def lib_count_vectorizer(it, stop=True):
    from sklearn.feature_extraction.text import CountVectorizer
    def f(it):
        if stop:
            stop = 'english'
        else:
            stop = None
        v = CountVectorizer(it, stop_words=stop)
        return v.fit_transform(it)
    return f

def lib_hash_vectorizer(it, stop=True):
    from sklearn.feature_extraction.text import HashingVectorizer
    def f(it):
        if stop:
            stop = 'english'
        else:
            stop = None
        v = HashingVectorizer(it, stop_words=stop, non_negative=True, norm=None)
        return v.transform(it)
    return f
def my_dict_vectorizer(it, stop=True):
    import feature
    def f(it):
        v = feature.CountFeature(limit = -1, use_stopwords=stop)
        return v.transform(it)
    return f

class Main(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='CMPS242 project',
            usage='''main.py <method> [<options>] [--featue <vectorizer>]

The available methods::
            OneVsRest       [scratch]   Adapt problem into binary classificaiton
            LabelPowerset   [library]   Adapt multi-label into multi-class
            MLkNN           [library]   Multi label kNN

Available feature vectorizer:
    My_dictionary
    LIB_count
    LIB_hash
''')
        parser.add_argument('command', help='method to deal multi label')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print 'Unrecognized method'
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name

        self.sub_parser = argparse.ArgumentParser()
        # prefixing the argument with -- means it's optional
        self.sub_parser.add_argument('-f', default='My_dict',
                            choices=['My_dict', 'LIB_count', 'LIB_hash'])
        self.sub_parser.add_argument('-N', default=5000, type=int, help='Number of samples (train + test)')
        self.sub_parser.add_argument('-D', default=500, type=int, help='Number of labels ')
        self.sub_parser.add_argument('--stop', default=True, action='store_true', help='Use stop-words')
        self.sub_parser.add_argument('--per', default=0.8, type=float, help='percentage of train data')

        getattr(self, args.command)()

    def OneVsRest(self):
        #parser = argparse.ArgumentParser()
        # prefixing the argument with -- means it's optional
        self.sub_parser.add_argument('--library', action='store_true', default=False)
        self.sub_parser.add_argument('-c', default='My_NaiveBayes',
                            choices=['My_NaiveBayes', 'My_Logistic', 'LIB_NB', 'LIB_LR', 'LIB_SVM'],
                            help='binary classifier')

        args = self.sub_parser.parse_args(sys.argv[2:])
        if args.library:
            from sklearn.multiclass import OneVsRestClassifier
            ensembler = OneVsRestClassifier
            if args.c == 'My_NaiveBayes':
                print "[!] Error"
                print "    can't use our classifier with library's ensembler"
        else:
            ensembler = OneVsRest

        if args.c == 'My_NaiveBayes':
            classifier = NaiveBayes
        elif args.c == 'LIB_NB':
            from sklearn.naive_bayes import MultinomialNB
            classifier = MultinomialNB()
        elif args.c == 'LIB_SVM':
            from sklearn.svm import LinearSVC
            classifier = LinearSVC()

        if args.f == 'My_dict':
            vectorizer = my_dict_vectorizer(args.stop)
        elif args.f == 'LIB_count':
            vectorizer = lib_count_vectorizer(args.stop)
        elif args.f == 'LIB_hash':
            vectorizer = lib_hash_vectorizer(args.stop)

        print 'Running OneVsRest, arguments=%s' % args
        print 'Loading %s data...' %args.N
        data = load_data(args.N, args.D, args.per, vectorizer)
        print 'Done loading data, actual feature size:', data[1].shape
        print 'Running OneVsRest(%s) with %s' %("libray" if args.library else 'ours', args.c)
        run_OneVsRest(data, ensembler, classifier)

    def fetch(self):
        parser = argparse.ArgumentParser(
            description='Download objects and refs from another repository')
        # NOT prefixing the argument with -- means it's not optional
        parser.add_argument('repository')
        args = parser.parse_args(sys.argv[2:])
        print 'Running git fetch, repository=%s' % args.repository


if __name__ == '__main__':
    Main()

exit()
print "CMPS242 project"
print "Loading data"
# TODO: modify this:
n_samples = 60000
n_labels = 500
per = 0.6
vectorizer = my_dict_vectorizer
data = load_data(n_samples, n_labels, per, my_dict_vectorizer)

X, Y, Xt, Yt, cat = data
print type(X), type(Y)
#print "run OneVsRest with our implementation"
run_OneVsRest(data, OneVsRest, NaiveBayes)

print "run OneVsrest with sklearn"
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
print ">> Multinomial Naive Bayes"
run_OneVsRest(data, OneVsRestClassifier, MultinomialNB())
print ">> LinearSVC"
run_OneVsRest(data, OneVsRestClassifier, LinearSVC())
