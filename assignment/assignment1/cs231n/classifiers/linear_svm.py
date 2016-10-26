import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # X[i] is the i-th example
    # y is the arrary of label which is range(10)
    # y[i] is i-th example's label,
    # so scores[y[i]] is the current score of correct class
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss. L2 regulation, penalize the W
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # W is a (D, C) np.array

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  # X is (N,D), W is (D, C)
  scores = X.dot(W) # (N, C)
  correct_class_scores = scores[range(scores.shape[0]), y].reshape(-1, 1)

  # The influnce of numberic caculation !!!#
  # Here I hope to use the operations of additon and subtraction to set
  # margins[range(scores.shape[0]), y] to be zeros. It's OK to compute the loss,
  # but It has a very bad effect on the operation of [margins > 0],
  # because there are still some elements of margins[range(scores.shape[0]), y]
  # are very very small float numbers, and they are greater than zero!!!
  # scores[range(scores.shape[0]), y] -= 1 # set delta = 1

  margins = scores - correct_class_scores + 1
  # print margins[range(scores.shape[0]), y]

  margins[range(scores.shape[0]), y] = 0

  loss = margins[margins>0].sum() / X.shape[0]
  loss += 0.5 * reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  # the shape of dW is same to W (D, C)
  # scores is (N, C), margin is (N, C), x is (N, D)
  N = X.shape[0]
  #counts = np.zeros(margins.shape)
  #counts[margins>0] = 1
  counts = (margins>0) * 1
  counts[range(N), y] = -np.sum(counts, axis = 1)
  dW += X.T.dot(counts)

  dW /= N
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
