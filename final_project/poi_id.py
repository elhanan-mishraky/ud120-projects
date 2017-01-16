#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'total_stock_value',
                 'total_payments',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'to_poi_normalized',
                 'from_poi_normalized'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']
### Task 3: Create new feature(s)
# 'from_poi_to_this_person'/'from_messages'
# 'from_poi_to_this_person'/'to_messages'


def nan_to_zero(val):
    if val == "NaN":
        return 0
    else:
        return val


for key, value in data_dict.iteritems():
    from_this_person_to_poi = nan_to_zero(value['from_this_person_to_poi'])
    from_messages = nan_to_zero(value['from_messages'])
    to_poi_normalized = (float(from_this_person_to_poi) / from_messages) if from_messages != 0 else 0
    value['to_poi_normalized'] = to_poi_normalized

    from_poi_to_this_person = nan_to_zero(value['from_poi_to_this_person'])
    to_messages = nan_to_zero(value['to_messages'])
    from_poi_normalized = (float(from_poi_to_this_person) / to_messages) if from_messages != 0 else 0
    value['from_poi_normalized'] = from_poi_normalized

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

# features_train = features_train[:25]
# labels_train   = labels_train[:25]

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



# clf = AdaBoostClassifier()
# clf = RandomForestClassifier()
# Accuracy: 0.84567	Precision: 0.28216	Recall: 0.10200	F1: 0.14983	F2: 0.11693
# Total predictions: 15000	True positives:  204	False positives:  519	False negatives: 1796	True negatives: 12481
# clf = KNeighborsClassifier()
# Accuracy: 0.87100	Precision: 0.57143	Recall: 0.13000	F1: 0.21181	F2: 0.15376
# Total predictions: 15000	True positives:  260	False positives:  195	False negatives: 1740	True negatives: 12805
# clf = GaussianNB())
# Accuracy: 0.85060	Precision: 0.36916	Recall: 0.17000	F1: 0.23280	F2: 0.19056
# Total predictions: 15000	True positives:  340	False positives:  581	False negatives: 1660	True negatives: 12419
# clf = svm.SVC())
# clf = tree.DecisionTreeClassifier()
# Accuracy: 0.80993	Precision: 0.29213	Recall: 0.29900	F1: 0.29553	F2: 0.29760
# Total predictions: 15000	True positives:  598	False positives: 1449	False negatives: 1402	True negatives: 11551
clf = tree.DecisionTreeClassifier(criterion="entropy")
# Accuracy: 0.80993	Precision: 0.29213	Recall: 0.29900	F1: 0.29553	F2: 0.29760
# Total predictions: 15000	True positives:  598	False positives: 1449	False negatives: 1402	True negatives: 11551

# from sklearn.model_selection import GridSearchCV
# tuned_parameters = {'min_samples_split':[2, 20]}
#
# scores = ['precision', 'recall', "f1"]
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=20,
#                        scoring='%s_macro' % score)
#     clf.fit(features_train, labels_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = labels_test, clf.predict(features_test)
#     print(classification_report(y_true, y_pred))
#     print()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.Stratif iedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)