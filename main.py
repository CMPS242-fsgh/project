#!/usr/bin/env python2

import argparse
import sys
import time
import warnings

import loader
from feature import CountFeature
from classifier import (NaiveBayes, LogisticRegression)
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
def load_data(n, n_feature, n_test, vectorizer):
    N = n_test + n
    YY, categories = loader.get_target(n_feature, N)
    XX = vectorizer(loader.review_from_file(N))

    X, Y, Xt, Yt = loader.split_training(XX, YY, n_test)
    return X, Y, Xt, Yt, categories

@timeit
def run_OneVsRest(data, ensembler, classifier):
    X, Y, Xt, Yt, cats = data
    model = ensembler(classifier)
    model.fit(X, Y)
    Yp = model.predict(Xt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hl = computeMetrics(Yp, Yt, cats)

    print 'the hamming loss:'
    print '>>  ', hl
    from sklearn.metrics import (hamming_loss, classification_report)
    print 'hamming loss(library):', hamming_loss(Yt, Yp)
    print classification_report(Yt, Yp, target_names = cats)
    print 'DONE..'

@timeit
def run_LabelPowerset(data, ensembler, classifier):
    X, Y, Xt, Yt, cats = data
    model = ensembler(classifier, require_dense=[False, False])
    model.fit(X, Y)
    Yp = model.predict(Xt)
    #print '----'
    #print Yp.shape, Yt.shape
    if hasattr(Yp, 'toarray'):
        Yp = Yp.toarray()
    #print Yt, type(Yt)
    #print Yp, type(Yp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hl = computeMetrics(Yp, Yt, cats)

    print 'the hamming loss:'
    print '>>  ', hl
    print 'DONE..'
    from sklearn.metrics import (hamming_loss, classification_report)
    print 'hamming loss(library):', hamming_loss(Yt, Yp)
    print classification_report(Yt, Yp, target_names = cats)


def lib_count_vectorizer(stop=True, bigram=False):
    from sklearn.feature_extraction.text import CountVectorizer
    ngram = (1,2) if bigram else (1,1)
    stopf = stop
    def f(it):
        if stopf:
            stop = 'english'
        else:
            stop = None
        v = CountVectorizer(it, stop_words=stop, ngram_range=ngram)
        print stop
        return v.fit_transform(it)
    return f

def lib_hash_vectorizer(stop=True, bigram=False):
    from sklearn.feature_extraction.text import HashingVectorizer
    ngram = (1,2) if bigram else (1,1)
    stopf = stop
    def f(it):
        if stopf:
            stop = 'english'
        else:
            stop = None
        v = HashingVectorizer(it, stop_words=stop, non_negative=True, norm=None, ngram_range=ngram)
        return v.transform(it)
    return f

def lib_tfidf_vectorizer(stop=True, bigram=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    def f(it):
        v = TfidfVectorizer()
        return v.fit_transform(it)
    return f


def my_dict_vectorizer(stop=True, bigram=False):
    import feature
    stopf = stop
    def f(it):
        v = feature.CountFeature(limit = -1, use_stopwords=stopf, bigram=bigram)
        print stopf
        return v.transform(it)
    return f


class Main(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='CMPS242 project',
            usage='''main.py <method> [<options>] [--featue <vectorizer>]

1. Choose one of methods::
        OneVsRest                 Adapt problem into binary classificaiton
        LabelPowerset             Adapt multi-label into multi-class
        MLkNN                     Multi label kNN
        --library     [optional]  Use library implementation instead of ours

2. Choose one of feature vecterizer (ues -f)
        -f My_dict                Our implementation of dictionary vectorizer
        -f LIB_count              Counting vectorizer from sklearn
        -f LIB_hash               Hashing vectorizer from sklearn
        -f LIB_tfidf              Tf-idf
        --nostop                  Do not use stopwords
        --bigram                  Use bigram instead of unigram

3. Control the data size
        -N       [default 50000]  size of training
        -Nt      [default 10000]  size of testing
        -D       [default 100]   size of labels

4. Classifier options for OneVsRest:
        -c My_NaiveBayes
        -c My_Logistic
        -c LIB_NB
        -c LIB_LR
        -c LIB_SVM

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
                            choices=['My_dict', 'LIB_count', 'LIB_hash', 'LIB_tfidf'])
        self.sub_parser.add_argument('-N', default=5000, type=int, help='Number of train samples')
        self.sub_parser.add_argument('-D', default=500, type=int, help='Number of labels ')
        self.sub_parser.add_argument('--nostop', default=False, action='store_true', help='Do not use stop-words')
        self.sub_parser.add_argument('-Nt', default=10000, type=int, help='Number of Test data')
        self.sub_parser.add_argument('--bigram', default=False, action='store_true', help='Use bigram')

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
            if args.library:
                classifier = MultinomialNB()
            else:
                classifier = MultinomialNB
        elif args.c == 'My_Logistic':
            classifier = LogisticRegression
        elif args.c == 'LIB_LR':
            from sklearn.linear_model import LogisticRegression as LR
            if args.library:
                classifier = LR()
            else:
                classifier = LR

        elif args.c == 'LIB_SVM':
            from sklearn.svm import LinearSVC
            if args.library:
                classifier = LinearSVC()
            else:
                classifier = LinearSVC

        if args.f == 'My_dict':
            vectorizer = my_dict_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_count':
            vectorizer = lib_count_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_hash':
            vectorizer = lib_hash_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_tfidf':
            vectorizer = lib_tfidf_vectorizer(stop=not args.nostop, bigram=args.bigram)

        print 'Running OneVsRest, arguments=%s' % args
        print 'Loading %s data...' %args.N
        data = load_data(args.N, args.D, args.Nt, vectorizer)
        print 'Done loading data, actual feature size:', data[1].shape, 'X shape', data[0].shape
        print 'Running OneVsRest(%s) with %s' %("libray" if args.library else 'ours', args.c)
        run_OneVsRest(data, ensembler, classifier)

    def LabelPowerset(self):
        self.sub_parser.add_argument('--library', action='store_true', default=False)
        self.sub_parser.add_argument('-c', default='My_NaiveBayes',
                            choices=['My_NaiveBayes', 'My_Logistic', 'LIB_NB', 'LIB_LR', 'LIB_SVM'],
                            help='binary classifier')

        args = self.sub_parser.parse_args(sys.argv[2:])

        if args.library:
            from skmultilearn.problem_transform import (BinaryRelevance, LabelPowerset)
            ensembler = LabelPowerset
        else:
            from multi import LabelPowerset
            #from multi import LabelPowerSetClassifier
            ensembler = LabelPowerset
            #ensembler = LabelPowerSetClassifier


        if args.c == 'My_NaiveBayes':
            print "Not supported"
            exit()

        elif args.c == 'LIB_NB':
            from sklearn.naive_bayes import MultinomialNB
            if args.library:
                classifier = MultinomialNB()
            else:
                classifier = MultinomialNB

        elif args.c == 'LIB_SVM':
            from sklearn.svm import LinearSVC
            if args.library:
                classifier = LinearSVC()
            else:
                classifier = LinearSVC
        if args.f == 'My_dict':
            vectorizer = my_dict_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_count':
            vectorizer = lib_count_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_hash':
            vectorizer = lib_hash_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_tfidf':
            vectorizer = lib_tfidf_vectorizer(stop=not args.nostop, bigram=args.bigram)


        print 'Running Label Powerset, arguments=%s' % args
        print 'Loading %s data...' %args.N
        data = load_data(args.N, args.D, args.Nt, vectorizer)
        print 'Done loading data, actual feature size:', data[1].shape
        run_LabelPowerset(data, ensembler, classifier)
        print "OK"

    def MLkNN(self):
        self.sub_parser.add_argument('--library', action='store_true', default=False)

        args = self.sub_parser.parse_args(sys.argv[2:])
        print 'Running ML-kNN, arguments=%s' % args
        print 'Loading %s data...' %args.N

        if args.f == 'My_dict':
            vectorizer = my_dict_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_count':
            vectorizer = lib_count_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_hash':
            vectorizer = lib_hash_vectorizer(stop=not args.nostop, bigram=args.bigram)
        elif args.f == 'LIB_tfidf':
            vectorizer = lib_tfidf_vectorizer(stop=not args.nostop, bigram=args.bigram)



        data = load_data(args.N, args.D, args.Nt, vectorizer)
        print 'Done loading data, actual feature size:', data[1].shape

        X, Y, Xt, Yt, cats = data
        if args.library:
            from skmultilearn.adapt import MLkNN
            model = MLkNN()
        else:
            from sklearn.neighbors import NearestNeighbors
            from multi import MLkNN
            model = MLkNN(NearestNeighbors)
        model.fit(X, Y)
        Yp = model.predict(Xt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hl = computeMetrics(Yp, Yt, cats)

        print 'the hamming loss:'
        print '>>  ', hl
        from sklearn.metrics import (hamming_loss, classification_report)
        print 'hamming loss(library):', hamming_loss(Yt, Yp)
        print classification_report(Yt, Yp, target_names = cats)
        print 'DONE..'



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
