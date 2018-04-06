#one-vs-all
import numpy as np
from ex2.gradientFunctionReg import gradientFunctionReg
from ex3.lrCostFunction import lrCostFunction

theta_t = np.array([[-2.],[-1.], [1.], [2.]])
X_tem = (np.arange(1,16).reshape(5,3,order='F'))/10.
X_t = np.column_stack((np.ones((5, 1)), X_tem))
y_t = np.array([[1.],[0.], [1.], [0.], [1.]])
lambda_t = 3
cost_t = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad_t = gradientFunctionReg(theta_t, X_t, y_t, lambda_t)
print 'Test Case 1, cost: %4f' % cost_t 
print 'Test Case 1, grad: ' ,["%0.4f" % i for i in grad_t] 