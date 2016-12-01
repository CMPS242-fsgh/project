import numpy as np

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)
    
def computeMetrics(predicted, test,label_set):
    #predicted = np.array([[1, 0, 0, 0,1], [0,0,1,1,1],[0,0,1,1,1],[1,1,1,1,1],[0,0,0,1,1],[0,1,1,1,0]], np.int32)
    #test = np.array([[1, 0, 1, 0,1], [0,1,1,0,1],[01,1,1,1,1],[1,0,0,0,1],[0,1,0,1,1],[1,1,0,0,0]], np.int32)  
    #label_set = ["A","b","c","d","e"]
    CM = np.zeros((2,2))
    HL = 0 
    test_sample_size = predicted.shape[0]
    n_categories = len(label_set)
    for i in range( n_categories ):
        print  label_set[i]
        predicted_vector = predicted[:,i]
        test_vector = test[:,i]
        for x, y in np.nditer([predicted_vector,test_vector]): 
            a = int(x)
            b = int(y)
            CM[a][b] = CM[a][b] + 1
    #print type(yt) is np.ndarray
        print CM
        precision = CM[1][1] / (CM[1][1]+CM[1][0])
        recall = CM[1][1] / (CM[1][1]+CM[0][ 1])
        F1_score  =  2 * (precision * recall) / (precision + recall)
        print F1_score
        HL = HL + CM[0][1]+CM[1][0]
        CM = np.zeros((2,2))
    HL = HL/(test_sample_size*n_categories)
    print HL
    print hamming_score(predicted,test)
    return HL

