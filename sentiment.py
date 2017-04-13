import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random

from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    #English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    whole = train_pos + train_neg
    total =set()

    for review in whole:
        for word in review:
            if(word not in stopwords):
                total.add(word)


    #print len(total)
    """
    train_pos_set = set()
    for review in train_pos:
        for word in review:
            train_pos_set.add(word)

    train_pos_set = train_pos_set.difference(stopwords)
    #print train_pos_set

    train_neg_set = set()
    for review in train_neg:
        for word in review:
            train_neg_set.add(word)

    train_neg_set = train_neg_set.difference(stopwords)

    merged_set = train_pos_set.union(train_neg_set)
    #print merged_set

    """

    pos_dict = dict();
    for review in train_pos:
        for word in set(review):
            if(pos_dict.has_key(word)):
                pos_dict[word] = pos_dict[word] + 1
            else:
                pos_dict[word] = 1

    #print len(pos_dict)

    neg_dict = dict();
    for review in train_neg:
        for word in set(review):
            if(neg_dict.has_key(word)):
                neg_dict[word] = neg_dict[word] + 1
            else:
                neg_dict[word]=1

    #print len(neg_dict)
    #merged_list = list(merged_set)
    merged_list = list(total)
    #print len(merged_list)
    pos_neg_list = []
    

    for word in merged_list:
        count_pos=0
        count_neg=0
        if(pos_dict.has_key(word)):
            count_pos = int(pos_dict.get(word)) 
        if(neg_dict.has_key(word)):
            count_neg = int(neg_dict.get(word))
        if((count_pos>=(0.01*len(train_pos)) or count_neg>=(0.01*len(train_neg))) and (count_pos>=(2*count_neg) or count_neg>=(2*count_pos))):
            #merged_list.remove(word)
            pos_neg_list.append(word)

    #print "length:"
    #print len(pos_neg_list)

    """
    #print merged_list
    merged_list = []
    final_list = []
    for word in pos_neg_list:
        count_pos=0
        count_neg=0
        if(pos_dict.has_key(word)):
            count_pos = int(pos_dict.get(word))
        if(neg_dict.has_key(word)):
            count_neg = int(neg_dict.get(word))
        if((count_pos>=2*count_neg or count_neg>=2*count_pos)):
            final_list.append(word)

    """
    #print "length:"
    #print len(final_list)

    l1 = []
    #l2 = []
    train_pos_vec = []
    train_neg_vec = []
    test_neg_vec = []
    test_pos_vec = []

    

    for review in train_pos:
        l1 = []
        for word in pos_neg_list:
            if word in review:
                l1.append(1);
            else:
                l1.append(0);
        train_pos_vec.append(l1)

    #print len(train_pos_vec)
    
    #l1 = []
    #l2 = []
    #train_pos_vec = create_vector(train_pos,pos_neg_list)

    #print train_pos_vec

    for review in train_neg:
        l1 = []
        for word in pos_neg_list:
            if word in review:
                l1.append(1);
            else:
                l1.append(0);
        train_neg_vec.append(l1)

    #l1 = []
    #l2 = []

    for review in test_pos:
        l1 = []
        for word in pos_neg_list:
            if word in review:
                l1.append(1)
            else:
                l1.append(0)
        test_pos_vec.append(l1)

    #l1 = []
    #l2 = []

    for review in test_neg:
        l1 = []
        for word in pos_neg_list:
            if word in review:
                l1.append(1);
            else:
                l1.append(0);
        test_neg_vec.append(l1)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec
"""
def create_vector(l,pos_neg_list):
    review_vector = []
    l1 = []
    for review in l:
        l1 = []
        for word in pos_neg_list:
            if word in review:
                l1.append(1)
            else:
                l1.append(0)
        review_vector.append(l1)

    print review_vector
    return review_vector
"""
def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    i=0

    labeled_train_pos = []
    for review in train_pos: 
        labeled_train_pos.append(LabeledSentence(words=review, tags=["train_pos_"+str(i)]))
        i=i+1

    len_train_pos = i
    
    i=0

    labeled_train_neg = []
    for review in train_neg: 
        labeled_train_neg.append(LabeledSentence(words=review, tags=["train_neg_"+str(i)]))
        i=i+1

    len_train_neg = i

    i=0

    labeled_test_pos = []
    for review in test_pos: 
        labeled_test_pos.append(LabeledSentence(words=review, tags=["test_pos_"+str(i)]))
        i=i+1


    len_test_pos = i

    i=0

    labeled_test_neg = []
    for review in test_neg: 
        labeled_test_neg.append(LabeledSentence(words=review, tags=["test_neg_"+str(i)]))
        i=i+1


    len_test_neg = i

    #print len(labeled_train_pos)
    #print len(labeled_train_neg)
    #print len(labeled_test_pos)
    #print len(labeled_test_neg)

    #length of all is same
    #no_of_loops = i

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = []

    for x in range(0, len_train_pos):
        r_tag = "train_pos_"+str(x)
        train_pos_vec.append(model.docvecs[r_tag])

    train_neg_vec = []

    for x in range(0, len_train_neg):
        r_tag = "train_neg_"+str(x)
        train_neg_vec.append(model.docvecs[r_tag])

    test_pos_vec = []

    for x in range(0, len_test_pos):
        r_tag = "test_pos_"+str(x)
        test_pos_vec.append(model.docvecs[r_tag])

    test_neg_vec = []

    for x in range(0, len_test_neg):
        r_tag = "test_neg_"+str(x)
        test_neg_vec.append(model.docvecs[r_tag])

    #print train_pos_vec
    #print train_neg_vec
    #print test_pos_vec
    #print test_neg_vec
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = BernoulliNB()
    nb_model = nb_model.fit(train_pos_vec + train_neg_vec,Y)
    BernoulliNB(alpha=1.0,binarize=None)

    lr_model = LogisticRegression()
    lr_model = lr_model.fit(train_pos_vec+train_neg_vec,Y)
    LogisticRegression()

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = GaussianNB()
    nb_model = nb_model.fit(train_pos_vec+train_neg_vec, Y)
    lr_model = LogisticRegression()
    lr_model = lr_model.fit(train_pos_vec+train_neg_vec,Y)
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    prediction = model.predict(test_pos_vec)

    tp = 0
    fn = 0
    for s in prediction:
        if s == "pos":
            tp = tp+1
        else:
            fn = fn+1

    prediction = model.predict(test_neg_vec)

    tn = 0
    fp = 0
    for s in prediction:
        if s == "neg":
            tn = tn+1
        else:
            fp = fp+1

    accuracy = float((tn+tp))/float(tn+tp+fn+fp)

    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
