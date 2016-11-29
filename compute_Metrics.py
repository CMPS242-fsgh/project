import numpy as np

def computeMetrics(predicted, test,label_set=None):
    #predicted = np.array([[1, 0, 0, 0,1], [0,0,1,1,1],[0,0,1,1,1],[1,1,1,1,1],[0,0,0,1,1],[0,1,1,1,0]], np.int32)
    #test = np.array([[1, 0, 1, 0,1], [0,1,1,0,1],[01,1,1,1,1],[1,0,0,0,1],[0,1,0,1,1],[1,1,0,0,0]], np.int32)
    #label_set = ["A","b","c","d","e"]
    CM = np.zeros((2,2))
    HL = 0
    test_sample_size = predicted.shape[0]
    n_categories = predicted.shape[1]
    for i in range( n_categories ):
        #print  label_set[i]
        predicted_vector = predicted[:,i]
        test_vector = test[:,i]
        for x, y in np.nditer([predicted_vector,test_vector]):
            a = int(x)
            b = int(y)
            CM[a][b] = CM[a][b] + 1
            #print type(yt) is np.ndarray
        #print CM
        precision = CM[1][1] / (CM[1][1]+CM[1][0])
        recall = CM[1][1] / (CM[1][1]+CM[0][ 1])
        F1_score  =  2 * (precision * recall) / (precision + recall)
        #print F1_score
        HL = HL + CM[0][1]+CM[1][0]
        CM = np.zeros((2,2))

    HL = HL/(test_sample_size*n_categories)
    print HL
    return HL
