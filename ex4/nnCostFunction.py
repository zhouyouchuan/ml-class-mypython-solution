import numpy as np

from ex2.sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy()



# Setup some useful variables
    m, _ = X.shape


# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#
    X = np.column_stack((np.ones((m, 1)), X))
    z2 = np.dot(X,Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    #htheta = np.argmax(a3, axis=1)
    
    yVec = np.zeros(m*num_labels).reshape(m,num_labels)
    
    for i in range(0,num_labels):
        y_tem = np.where(y == i+1,1,0)
        yVec[:,i] = y_tem
    
    ureg_J = (-yVec) * (np.log(a3)) - (1 - yVec) * (np.log(1 - a3))
    ureg_J = (1./m) *np.sum(ureg_J)
    
    ureg_T1 = Theta1[:,1:]
    ureg_T2 = Theta2[:,1:]
    J = ureg_J + (Lambda/(2.*m)) * (np.sum(np.power(ureg_T1,2)) + np.sum(np.power(ureg_T2,2)))
    
    DELTA2 = np.zeros((hidden_layer_size, input_layer_size+1 ))
    DELTA3 = np.zeros((num_labels, hidden_layer_size+1 ))
    for i in range(m):
        a1_s = X[i,:]
        z2_s = np.dot(a1_s, Theta1.T)
        a2_s = sigmoid(z2_s)
        a2_s = np.insert(a2_s,0,1)
        z3_s = np.dot(a2_s, Theta2.T)
        a3_s = sigmoid(z3_s)
        delta3 = a3_s - yVec[i,:]
        delta2 = np.dot(delta3, ureg_T2) * sigmoidGradient(z2_s)
        delta2 = delta2.reshape(hidden_layer_size, 1)
        delta3 = delta3.reshape(num_labels, 1)
        DELTA2 = DELTA2 + np.dot(delta2.reshape(hidden_layer_size, 1), a1_s.reshape(input_layer_size+1, 1).T)
        DELTA3 = DELTA3 + np.dot(delta3.reshape(num_labels, 1),a2_s.reshape(hidden_layer_size+1, 1).T)
    
    Theta1_grad = (1./m) * DELTA2
    Theta2_grad = (1./m) * DELTA3
    #a = Theta1_grad[:,1:]
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (1.0*Lambda/m)*ureg_T1
    #b = Theta1_grad[:,1:]
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (1.0*Lambda/m)*ureg_T2
    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradient
    grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))


    return J, grad