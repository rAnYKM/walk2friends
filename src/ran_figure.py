import pandas as pd

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def draw_pr_curve(y_test, y_score):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.show()

def load_test_scores(city, model_name, walk_len=100, walk_times=20,
                num_features=128):

    feature = pd.read_csv('dataset/'+city+'/feature/'+city+'_'+model_name+'_'+\
                         str(int(walk_len))+'_'+str(int(walk_times))+'_'+str(int(num_features))+'.feature',\
                          names = ['u1', 'u2', 'label',
                                   'cosine', 'euclidean', 'correlation', 'chebyshev',\
                                   'braycurtis', 'canberra', 'cityblock', 'sqeuclidean'])

    for i in ['cosine']:
        max_score = max(feature[i])
        new_feature = [1 - item/max_score for item in feature[i]]
        draw_pr_curve(feature.label, new_feature)