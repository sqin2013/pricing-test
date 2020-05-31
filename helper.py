import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize = (6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
	

def plot_barh(feature_importance, xlabel='Importance', ylabel='Feature'):
	"""
	plot horizontal bar chart
	
	parameters
	----------
	feature_importance: pandas series. index for labels, values for bar height
	"""
	s = feature_importance.sort_values(ascending=True)
	fig, ax = plt.subplots()
	labels = s.index
	y = np.arange(len(labels))
	ax.barh(y, s.values)
	ax.set_yticks(y)
	ax.set_yticklabels(labels)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
