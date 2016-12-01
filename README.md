CMPS242 Project

#Usage:

1. Choose one of methods:

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


#Archievements:

1. Code from scratch(Depend on scipy only):

* OneVsRest algorithm

* Label Powerset

* Multi Label k Nearest Neighbor

* Naive Bayes binary classifier

* Logistic Regression binary classifier

2. Better performance than Naive Bayes implementation in sklearn

3. Aglorithm works entire yelp data (4 GiB)
