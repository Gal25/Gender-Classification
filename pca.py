from string import digits

import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment
import os, nltk, random, json
from nltk import word_tokenize
from nltk.classify import apply_features, SklearnClassifier, maxent
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from operator import itemgetter
import getpass
import numpy as np
import pickle
import datetime
import time
import seaborn as sns; sns.set()
import random as rd
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# INITIAL FUNCTIONS
#############################################################
def optimizemodel_sc(train_set2, labels_train_set2, test_set2, labels_test_set2, modelname, classes, testing_set,
                     selectedfeature, training_data):
    filename = modelname
    start = time.time()

    c1 = 0
    c5 = 0

    try:
        # decision tree
        classifier2 = DecisionTreeClassifier(random_state=0)
        classifier2.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier2, test_set2, labels_test_set2, cv=5)
        print('Decision tree accuracy (+/-) %s' % (str(scores.std())))
        c2 = scores.mean()
        c2s = scores.std()
        print(c2)
    except:
        c2 = 0
        c2s = 0

    try:
        classifier3 = GaussianNB()
        classifier3.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier3, test_set2, labels_test_set2, cv=5)
        print('Gaussian NB accuracy (+/-) %s' % (str(scores.std())))
        c3 = scores.mean()
        c3s = scores.std()
        print(c3)
    except:
        c3 = 0
        c3s = 0

    try:
        # svc
        classifier4 = SVC()
        classifier4.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier4, test_set2, labels_test_set2, cv=5)
        print('SKlearn classifier accuracy (+/-) %s' % (str(scores.std())))
        c4 = scores.mean()
        c4s = scores.std()
        print(c4)
    except:
        c4 = 0
        c4s = 0

    try:
        # adaboost
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier6, test_set2, labels_test_set2, cv=5)
        print('Adaboost classifier accuracy (+/-) %s' % (str(scores.std())))
        c6 = scores.mean()
        c6s = scores.std()
        print(c6)
    except:
        c6 = 0
        c6s = 0

    try:
        # gradient boosting
        classifier7 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier7, test_set2, labels_test_set2, cv=5)
        print('Gradient boosting accuracy (+/-) %s' % (str(scores.std())))
        c7 = scores.mean()
        c7s = scores.std()
        print(c7)
    except:
        c7 = 0
        c7s = 0

    try:
        # logistic regression
        classifier8 = LogisticRegression(random_state=1)
        classifier8.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier8, test_set2, labels_test_set2, cv=5)
        print('Logistic regression accuracy (+/-) %s' % (str(scores.std())))
        c8 = scores.mean()
        c8s = scores.std()
        print(c8)
    except:
        c8 = 0
        c8s = 0

    try:
        # voting
        classifier9 = VotingClassifier(
            estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
        classifier9.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier9, test_set2, labels_test_set2, cv=5)
        print('Hard voting accuracy (+/-) %s' % (str(scores.std())))
        c9 = scores.mean()
        c9s = scores.std()
        print(c9)
    except:
        c9 = 0
        c9s = 0

    try:
        # knn
        classifier10 = KNeighborsClassifier(n_neighbors=7)
        classifier10.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier10, test_set2, labels_test_set2, cv=5)
        print('K Nearest Neighbors accuracy (+/-) %s' % (str(scores.std())))
        c10 = scores.mean()
        c10s = scores.std()
        print(c10)
    except:
        c10 = 0
        c10s = 0

    try:
        # randomforest
        classifier11 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        classifier11.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier11, test_set2, labels_test_set2, cv=5)
        print('Random forest accuracy (+/-) %s' % (str(scores.std())))
        c11 = scores.mean()
        c11s = scores.std()
        print(c11)
    except:
        c11 = 0
        c11s = 0

    try:
        ##        #svm
        classifier12 = svm.SVC(kernel='linear', C=1.0)
        classifier12.fit(train_set2, labels_train_set2)
        scores = cross_val_score(classifier12, test_set2, labels_test_set2, cv=5)
        print('svm accuracy (+/-) %s' % (str(scores.std())))
        c12 = scores.mean()
        c12s = scores.std()
        print(c12)
    except:
        c12 = 0
        c12s = 0

    # IF IMBALANCED, USE http://scikit-learn.org/dev/modules/generated/sklearn.naive_bayes.ComplementNB.html

    maxacc = max([c2, c3, c4, c6, c7, c8, c9, c10, c11, c12])

    # if maxacc == c1:
    #     print('most accurate classifier is Naive Bayes' + 'with %s' % (selectedfeature))
    #     classifiername = 'naive-bayes'
    #     classifier = classifier1
    #     # show most important features
    #     classifier1.show_most_informative_features(5)
    if maxacc == c2:
        print('most accurate classifier is Decision Tree' + 'with %s' % (selectedfeature))
        classifiername = 'decision-tree'
        classifier2 = DecisionTreeClassifier(random_state=0)
        classifier2.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier = classifier2
    # elif maxacc == c3:
    #     print('most accurate classifier is Gaussian NB' + 'with %s' % (selectedfeature))
    #     classifiername = 'gaussian-nb'
    #     classifier3 = GaussianNB()
    #     classifier3.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
    #     classifier = classifier3
    # elif maxacc == c4:
    #     print('most accurate classifier is SK Learn' + 'with %s' % (selectedfeature))
    #     classifiername = 'sk'
    #     classifier4 = SVC()
    #     classifier4.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
    #     classifier = classifier4
    # elif maxacc == c5:
    #     print('most accurate classifier is Maximum Entropy Classifier' + 'with %s' % (selectedfeature))
    #     classifiername = 'max-entropy'
    #     classifier = classifier5
    # can stop here (c6-c10)
    elif maxacc == c6:
        print('most accuracate classifier is Adaboost classifier' + 'with %s' % (selectedfeature))
        classifiername = 'adaboost'
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier = classifier6
    elif maxacc == c7:
        print('most accurate classifier is Gradient Boosting ' + 'with %s' % (selectedfeature))
        classifiername = 'graidentboost'
        classifier7 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier = classifier7
    # elif maxacc == c8:
    #     print('most accurate classifier is Logistic Regression ' + 'with %s' % (selectedfeature))
    #     classifiername = 'logistic_regression'
    #     classifier8 = LogisticRegression(random_state=1)
    #     classifier8.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
    #     classifier = classifier8
    elif maxacc == c9:
        print('most accurate classifier is Hard Voting ' + 'with %s' % (selectedfeature))
        classifiername = 'hardvoting'
        classifier7 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifier7.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier8 = LogisticRegression(random_state=1)
        classifier8.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier6 = AdaBoostClassifier(n_estimators=100)
        classifier6.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier9 = VotingClassifier(
            estimators=[('gradboost', classifier7), ('logit', classifier8), ('adaboost', classifier6)], voting='hard')
        classifier9.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier = classifier9
    elif maxacc == c10:
        print('most accurate classifier is K nearest neighbors ' + 'with %s' % (selectedfeature))
        classifiername = 'knn'
        classifier10 = KNeighborsClassifier(n_neighbors=7)
        classifier10.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier = classifier10
    elif maxacc == c11:
        print('most accurate classifier is Random forest ' + 'with %s' % (selectedfeature))
        classifiername = 'randomforest'
        classifier11 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        classifier11.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
        classifier = classifier11
    # elif maxacc == c12:
    #     print('most accurate classifier is SVM ' + ' with %s' % (selectedfeature))
    #     classifiername = 'svm'
    #     classifier12 = svm.SVC(kernel='linear', C=1.0)
    #     # classifier12.fit(train_set2 + test_set2, labels_train_set2 + labels_test_set2)
    #     classifier = classifier12

    modeltypes = ['decision-tree', 'gaussian-nb', 'sk', 'adaboost', 'gradient boosting', 'logistic regression',
                  'hard voting', 'knn', 'random forest', 'svm']
    accuracym = [c2, c3, c4, c6, c7, c8, c9, c10, c11, c12]
    accuracys = [c2s, c3s, c4s, c6s, c7s, c8s, c9s, c10s, c11s, c12s]
    model_accuracy = list()
    for i in range(len(modeltypes)):
        model_accuracy.append([modeltypes[i], accuracym[i], accuracys[i]])

    model_accuracy.sort(key=itemgetter(1))
    endlen = len(model_accuracy)

    # print('saving classifier to disk')
    # f = open(modelname + '.pickle', 'wb')
    # pickle.dump(classifier, f)
    # f.close()

    end = time.time()

    execution = end - start

    print('summarizing session...')

    accstring = ''

    for i in range(len(model_accuracy)):
        accstring = accstring + '%s: %s (+/- %s)\n' % (
        str(model_accuracy[i][0]), str(model_accuracy[i][1]), str(model_accuracy[i][2]))

    training = len(train_set2)
    testing = len(test_set2)

    summary = 'SUMMARY OF MODEL SELECTION \n\n' + 'WINNING MODEL: \n\n' + '%s: %s (+/- %s) \n\n' % (
    str(model_accuracy[len(model_accuracy) - 1][0]), str(model_accuracy[len(model_accuracy) - 1][1]),
    str(model_accuracy[len(model_accuracy) - 1][2])) + 'MODEL FILE NAME: \n\n %s.pickle' % (
                  filename) + '\n\n' + 'DATE CREATED: \n\n %s' % (
                  datetime.datetime.now()) + '\n\n' + 'EXECUTION TIME: \n\n %s\n\n' % (
                  str(execution)) + 'GROUPS: \n\n' + str(classes) + '\n' + '('  + ' in each class, ' + str(
        int(testing_set * 100)) + '% used for testing)' + '\n\n' + 'TRAINING SUMMARY:' + '\n\n' + training_data + 'FEATURES: \n\n %s' % (
                  selectedfeature) + '\n\n' + 'MODELS, ACCURACIES, AND STANDARD DEVIATIONS: \n\n' + accstring + '\n\n' + '(C) 2018, NeuroLex Laboratories'

    data = {
        'model': modelname,
        'modeltype': model_accuracy[len(model_accuracy) - 1][0],
        'accuracy': model_accuracy[len(model_accuracy) - 1][1],
        'deviation': model_accuracy[len(model_accuracy) - 1][2]
    }

    return [model_accuracy[endlen - 1], summary, data]

def find_closest_to_zero(derivatives):
  min_diff = 0.0001 #good for 104
  min_index = None
  for i, d in enumerate(derivatives):
    diff = abs(d)
    if diff < min_diff:
        # min_diff = diff
        min_index = i
        break
  return derivatives[min_index], min_index

if __name__ == '__main__':
    jsonfilename_208 = open(r"\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\data\males_out_females_out_audio.json")
    jsonfilename_616 = open(r"C:\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\boys_girls_audio.json")
    jsonfilename_104 = open(r"C:\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\boys_girls_audio_104.json")
    jsonfilename_512 = open(r"C:\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\boys_girls_audio_512.json")
    data = json.load(jsonfilename_208)

    #select the data to featuc
    classes = list(data)
    features = list()
    labels = list()
    for i in range(len(classes)):
        for j in range(len(data[classes[i]])):
            feature = data[classes[i]][j]
            features.append(feature)
            labels.append(classes[i])

    # data_new = pd.DataFrame(columns=[*labels], index=features)
    # scale_data = preprocessing.scale(data.T)  # transpose the data. to have 1 and not 0
    # print(data_new)
    # print(len(labels))

    # print(data_new.shape)
    # if males - 0 females - 1
    lables_test = []
    for i in labels:
        if i == "males":
            lables_test.append(0)
        else:
            lables_test.append(1)
    # pca = PCA()
    # this is where the pca do all the calculation

##########################
    # sel = VarianceThreshold(threshold=0.99113985)
    # X = sel.fit_transform(features)
#########################

    # print(X.shape)
    pca = PCA()
    # pca.fit(data)    # generate coordinates for PCA graph based on the loading scores and scales data
    pca_data = pca.fit_transform(features)

    print(pca_data.shape)

    np_arr = np.asarray(features)
    covariance_matrix = np.cov(np_arr.T)

    # Get the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    derivatives = []
    for i in range(1, len(np.cumsum(pca.explained_variance_ratio_))):
        y2 = np.cumsum(pca.explained_variance_ratio_)[i]
        y1 = np.cumsum(pca.explained_variance_ratio_)[i - 1]
        x2 = i
        x1 = i - 1
        derivative = (y2 - y1) / (x2 - x1)
        derivatives.append(derivative)

    # print("derivatives", derivatives)
    n_com ,index = find_closest_to_zero(derivatives)
    n_components = np.cumsum(pca.explained_variance_ratio_)[index]
    print(np.cumsum(pca.explained_variance_ratio_)[index])

    sel = VarianceThreshold(threshold=n_components)
    X = sel.fit_transform(features)
    print(X.shape)


    # data_new_x = pd.DataFrame(columns=[*labels], index=sel)

    pca_1 = PCA(n_components=n_components)

    pca_data_1 = pca_1.fit_transform(features)

    # print(X.shape)
    print(pca_data_1.shape)

    # calculate the percentage of variation that each principal component account for
    # per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # lable = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    classnum = input('how many classes are you training?')

    folderlist = list()
    a = 0
    while a != int(classnum):
        folderlist.append(input('what is the folder name for class %s?' % (str(a + 1))))
        a = a + 1

    name = ''
    for i in range(len(folderlist)):
        if i == 0:
            name = name + folderlist[i]
        else:
            name = name + '_' + folderlist[i]

    modelname = name + '_sc_audio'
    testing_set = 0.33
    model_dir = os.getcwd() + '/models'

    # # date_tran = pca_data_1.transpose()
    # classes = list(pca_data_1)
    # features_2 = list()
    # labels_2 = list()
    # for i in range(len(classes)):
    #     for j in range(pca_data_1[classes[i]]):
    #         feature = pca_data_1[classes[i]][j]
    #         features_2.append(feature)
    #         labels_2.append(classes[i])

    train_set, test_set, train_labels, test_labels = train_test_split(pca_data_1,
                                                                      labels,
                                                                      test_size=testing_set,
                                                                      random_state=42)
    try:
        os.chdir(model_dir)
    except:
        os.mkdir(model_dir)
        os.chdir(model_dir)

    g = open(modelname + '_training_data.txt', 'w')
    g.write('train labels' + '\n\n' + str(train_labels) + '\n\n')
    g.write('test labels' + '\n\n' + str(test_labels) + '\n\n')
    g.close()

    training_data = open(modelname + '_training_data.txt').read()

    # MODEL OPTIMIZATION / SAVE TO DISK
    #################################################################
    selectedfeature = 'audio features (mfcc coefficients).'
    # min_num = len(X[classes[0]])
    [audio_acc, audio_summary, pca_data_1] = optimizemodel_sc(train_set, train_labels, test_set, test_labels,
                                                                     modelname, classes, testing_set,
                                                                     selectedfeature, training_data)

    g = open(modelname + '.txt', 'w')
    g.write(audio_summary)
    g.close()

    g2 = open(modelname + '.json', 'w')
    json.dump(data, g2)
    g2.close()

    # print(audio_model)
    print(audio_acc)





    # plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_lable=lable)
    # # plt.scatter(pca_data[:, 0], pca_data[:, 1], c=lables_test)
    # plt.xlabel('Principal Component')
    # plt.ylabel('Percentage of Explained Variance')
    # plt.title('Scree Plot')
    # plt.show(
    # x= 0
    # for i in np.cumsum(pca.explained_variance_ratio_):
    #     if i < 0.99113985:
    #         x+=1


    # print("var", np.cumsum(pca.explained_variance_ratio_))
    # print("x", x) # in 208 is 30 , in 616 is 161 , 104 is 26
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('pca_before')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    plt.plot(np.cumsum(pca_1.explained_variance_ratio_))
    plt.title('pca_after')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()