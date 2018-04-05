def plot_cm(clf, labels = ['rovnix', 'None', 'conficker', 'murofet', 'ramdo', 'tinba']):
    cm  = confusion_matrix(y, clf['y_pred'])
    percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T) 
    print ('\nConfusion Matrix Stats : ',clf['name'],clf['acc'],'%')
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print ("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))
            
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap=plt.cm.Blues)
    pylab.title('Confusion matrix of the classifier : '+ clf['name']+'\n')
    fig.colorbar(cax)
    ax.set_xticklabels([' '] + labels)
    ax.set_yticklabels([' '] + labels)
#    ax.text(0,0,percent[0][0],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
#    ax.text(0,1,percent[0][1],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
#    ax.text(1,0,percent[1][0],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
#    ax.text(1,1,percent[1][1],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    for x,Y in enumerate(percent):
        for a,b in enumerate(Y):
            ax.text(x,a,b,va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
            
    pylab.xlabel('Predicted')
    pylab.ylabel('Actual')
    pylab.savefig('./ClassifierImages/multi/'+clf['name']+'.png')
    pylab.show()
    
def plot_clf_cmpr(clf_array,labels = ['rovnix', 'None', 'conficker', 'murofet', 'ramdo', 'tinba']):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    dga_acc_matrix=[]
    for clf_pos,clf in enumerate(clf_array):
        cm  = confusion_matrix(y, clf['y_pred'])
        percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T) 
        dga_acc_matrix.append([max(x) for x in percent])
        for dga_pos,percent_array in enumerate(percent):
            ax.text(dga_pos,clf_pos,max(percent_array),va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))

    cax = ax.matshow(dga_acc_matrix, cmap=plt.cm.Blues)
    pylab.title('Confusion matrix of the classifier : '+ clf['name']+'\n')
    fig.colorbar(cax)
    ax.set_yticklabels([' '] + ['GNB','RFC','KNC','LSV'])
    ax.set_xticklabels([' '] + labels)
    pylab.xlabel('Predicted')
    pylab.ylabel('Actual')
#        pylab.savefig('./ClassifierImages/multi/'+clf['name']+'.png')
    print(clf['name'])
    pylab.show()
