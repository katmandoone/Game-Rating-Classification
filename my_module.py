import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import confusion_matrix_pretty_print as cmpp
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_model(hist):
    '''
    input: model history
    output: plots of accuracy and loss for train and val sets
    '''
    history = hist.history
    x = range(1, len(history['loss'])+1)
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].plot(x, history['acc'], label='acc')
    ax[0].plot(x, history['val_acc'], label='val_acc')
    ax[0].legend()
    ax[0].set_xticks(range(1,(len(history['acc']))+1),5)
    ax[0].set_title('Accuracy')
    ax[1].plot(x, history['loss'], label='loss')
    ax[1].plot(x, history['val_loss'], label='val_loss')
    ax[1].legend()
    ax[1].set_xticks(range(1,(len(history['loss']))+1),5)
    ax[1].set_title('Loss')
    plt.show()
    
def true_pred(model, test_gen):
    '''
    input: model, test_generator
    output: true classes, predicted classes
    '''
    y_true = test_gen.classes
    predictions = model.predict(test_gen)
    if predictions.shape[1] > 1:
    	y_pred = np.array([pred.argmax() for pred in predictions])
    else:
    	y_pred = np.round(predictions).astype('int').flatten()
    y_pred = predictions.round().flatten().astype('int')
    return y_true, y_pred

def get_labels(y_true, y_pred, test_gen):
    '''
    input: true classes, predicted classes, test generator
    output: class labels, true class labels, predicted class labels
    '''
    labels = [x for x,y in test_gen.class_indices.items()]
    true_labels = [labels[i] for i in y_true]
    pred_labels = [labels[i] for i in y_pred]
    return labels, true_labels, pred_labels

def show_cm(labels, y_true, y_pred):
    '''
    input: class labels, true classes, predicted classes
    output: confusion matrix
    '''
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, columns=labels, index=labels)
    cmpp.pretty_plot_confusion_matrix(cm_df, pred_val_axis='col', figsize=[7,7])
    cm = confusion_matrix(y_true, y_pred, labels, normalize='true')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
def pie_charts(true_lab, pred_lab, labels):
    '''
    input: true labels, predicted labels, class labels
    output: pie charts comparing distribution of true classes with predictions
    '''

    true_counts = [true_lab.count(x) for x in labels]
    pred_counts = [pred_lab.count(x) for x in labels]
    
    sns.set()
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].pie(true_counts, autopct='%.2f')
    ax[0].set_title('True')
    ax[1].pie(pred_counts, autopct='%.2f')
    ax[1].set_title('Predicted')
    fig.legend(labels=labels, loc='center')
    plt.suptitle('Distrubution of Classes')
    plt.tight_layout()

def evaluate(model, history, test_gen):
	''' evaluate model '''
	print(model.evalute(test_gen))
	y_true, y_pred = true_pred(model, test_gen)
	labels, true_labels, pred_labels = get_labels(y_true, y_pred, test_gen)
	plot_model(history)
	pie_charts(true_labels, pred_labels, labels)
	show_cm(labels, true_labels, pred_labels)
	print(classification_report(true_labels, pred_labels))
	plt.tight_layout()