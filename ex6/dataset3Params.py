import numpy as np
import sklearn.svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
    C = 1
    sigma = 0.3

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#
    C_vec = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    sig_vec = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    lenth = len(C_vec)
    pred_mat = np.ones((lenth,lenth))
    for i in range(lenth):
        for j in range(lenth):
            gamij = 1.0 / (2.0 * sig_vec[j] ** 2)
            clf = sklearn.svm.SVC(C=C_vec[i], kernel='rbf', tol=1e-3, max_iter=200, gamma=gamij)
            model = clf.fit(X,y)
            predictions = model.predict(Xval)
            pred_mat[i,j] = np.mean(np.double(predictions != yval))
    min_mat = np.argwhere(pred_mat == np.min(pred_mat))
    C_i = min_mat[0,0]
    sig_j = min_mat[0,1]
    C = C_vec[C_i]
    sigma = sig_vec[sig_j]
            
            

# =========================================================================
    return C, sigma
