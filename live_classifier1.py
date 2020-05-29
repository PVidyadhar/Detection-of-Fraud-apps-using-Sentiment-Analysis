import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class EnsembleClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Load all classifiers from the pickled files

# function to load models given filepath
def load_model(file_path):
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def get_sentiment(resut):
    


    # Original Naive Bayes Classifier
    ONB_Clf = load_model('pickled_algos/ONB_clf.pickle')

    # Multinomial Naive Bayes Classifier
    MNB_Clf = load_model('pickled_algos/MNB_clf.pickle')


    # Bernoulli  Naive Bayes Classifier
    BNB_Clf = load_model('pickled_algos/BNB_clf.pickle')

    # Logistic Regression Classifier
    LogReg_Clf = load_model('pickled_algos/LogReg_clf.pickle')

    # Stochastic Gradient Descent Classifier
    SGD_Clf = load_model('pickled_algos/SGD_clf.pickle')



    ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)
    # text='best'


    

    prediction=[]
    for i in result:
        feats = find_features(text)
        ans=ensemble_clf.classify(feats)
        temp=ensemble_clf.predict(i['content'])
        print(temp.astype(int))
        

        prediction.append(temp)

    p=0
    n=0
    for i in prediction:
        if i==1:
            p+=1
        else:
            n+=1
    overall=max(p,n)/(p+n)
    ratio=(p/(p+n))
    print(p,n,overall)
    fraud=0
    total=0
    li=['fraud','fake','duplicate','cheat','tricked','trap','deceive','scam']
    if(ratio<0.5):
        for r in result:
            rev=r['content']
            rev=rev.split(' ')
            print(rev)
            for i in rev:
                if i in li:
                    print(i)
                    fraud=1
                    break
            if fraud==1:
                total+=1
                fraud=0
    newfraud=0
    print(total)
    #if total>=((p+n)//2):
    if total >= 1:
        newfraud=1
    print(newfraud)
    return prediction,ratio,overall,newfraud

    #return (ans,ensemble_clf.confidence(feats))
#print(f("best"))
#server file open chei