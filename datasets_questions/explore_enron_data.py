#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# print enron_data["JAMES PRENTICE"][""]
# count = 0
# for key, value in enron_data.iteritems():
#     if value.get("poi"):
#         count += 1
# print count
count = 0
for key, value in enron_data.iteritems():
    if value.get("total_payments") == "NaN":
        count += 1
print count

count = 0
for key, value in enron_data.iteritems():
    if value.get("poi"):
        count += 1
print count

count = 0
for key, value in enron_data.iteritems():
    if value.get("poi") and value.get("total_payments") == "NaN":
        count += 1
print count

# LAY KENNETH L
# 103559793
# SKILLING JEFFREY K
# 8682716
# FASTOW ANDREW S
# 2424083