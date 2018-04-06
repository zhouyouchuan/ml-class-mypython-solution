# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
import numpy as np
import sklearn.svm
# ====================== GridSearchCV ======================
# class sklearn.model_selection.GridSearchCV(estimator, param_grid, \
# scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, \
# verbose=0, pre_dispatch=‘2*n_jobs’, error_score=’raise’, \
# return_train_score=’warn’)[source]
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
def modelSelecter(X,y,Xval,yval):
    C = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    sigma = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    gamma = 1.0 / (2.0 * np.power(sigma, 2))
    svc = sklearn.svm.SVC(kernel='rbf', class_weight='balanced',tol=1e-3, max_iter=200)
    parameters = {'C':C, 'gamma':gamma}
    #clf = sklearn.svm.SVC(C=C, kernel='rbf', tol=1e-3, max_iter=200, gamma=gamma)
    #svc = sklearn.svm.SVC() 
    clf = GridSearchCV(svc,parameters)
    clf = clf.fit(Xval,yval)
    y_pred = clf.predict(Xval)