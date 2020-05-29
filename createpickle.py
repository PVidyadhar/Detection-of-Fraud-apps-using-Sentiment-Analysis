import nltk
import os
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.classify import ClassifierI
from sklearn.metrics import f1_score, accuracy_score
# Defininig the ensemble model class 
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def create_pickle(c, file_name): 
    save_classifier = open(file_name, 'wb')
    pickle.dump(c, save_classifier)
    save_classifier.close()

files_pos = os.listdir('aclImdb/train/pos1')
files_pos = [open('aclImdb/train/pos1/'+f, 'r',encoding="utf8").read() for f in files_pos]
files_neg = os.listdir('aclImdb/train/neg1')
files_neg = [open('aclImdb/train/neg1/'+f, 'r',encoding="utf8").read() for f in files_neg]
all_words = []
documents = []

for p in  files_pos:
    
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "pos") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    pos = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in files_neg:
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "neg") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    neg = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


len(all_words)

save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
BOW = nltk.FreqDist(all_words)

word_features = list(BOW.keys())[:5000]

save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# Creating features for each review
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Shuffling the documents 
random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:20000]
testing_set = featuresets[20000:]
print( 'training_set :', len(training_set), '\ntesting_set :', len(testing_set))


print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set))*100)

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set))*100)

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set))*100)

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set))*100)

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_clf, testing_set))*100)

classifiers_dict = {'ONB': [classifier, 'pickled_algos/ONB_clf.pickle'],
                    'MNB': [MNB_clf, 'pickled_algos/MNB_clf.pickle'],
                    'BNB': [BNB_clf, 'pickled_algos/BNB_clf.pickle'],
                    'LogReg': [LogReg_clf, 'pickled_algos/LogReg_clf.pickle'],
                    'SGD': [SGD_clf, 'pickled_algos/SGD_clf.pickle'], 
                    'SVC': [SVC_clf, 'pickled_algos/SVC_clf.pickle']}




for clf, listy in classifiers_dict.items(): 
    create_pickle(listy[0], listy[1])

acc_scores = {}
for clf, listy in classifiers_dict.items(): 
    # getting predictions for the testing set by looping over each reviews featureset tuple
    # The first elemnt of the tuple is the feature set and the second element is the label 
    acc_scores[clf] = accuracy_score(ground_truth, predictions[clf])
    print(f'Accuracy_score {clf}: {acc_scores[clf]}')


