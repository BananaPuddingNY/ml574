import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    print('Dimensions of X: ', X.shape)
    k = int(np.max(y))
    d = np.shape(X)[1]
    means = np.empty((d, k))
    covmats = np.empty((d,d))
    for i in range(1, k + 1):
        A=[]
        for j in range(0, X.shape[0]):
            if y[j] == i:
               A.append(X[j,:])
        means[:,i-1] = np.mean(A,axis=0).transpose()
    covmat = np.cov(X, rowvar=0)
    print(means)
    print(covmat)
    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    k = int(np.max(y))
    d = np.shape(X)[1]
    means = np.empty((d, k))
    covmats = []
    for i in range(1, k + 1):
        A=[]
        for j in range(0, X.shape[0]):
            if y[j] == i:
                A.append(X[j, :])
        means[:, i-1] = np.mean(A, axis=0).transpose()
        covmatUnit = np.cov(A, rowvar=0)
        covmats.append(covmatUnit)
    print(means)
    print(covmats)
    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    Data_Num = np.shape(Xtest)[0]
    Class_Num = np.shape(means)[1]
    ypreds=[]
    match = 0
    for i in range(1,Data_Num+1):
        p_max = 0
        class_name = 0
        Test = np.transpose(Xtest[i-1,:])
        for j in range(1, Class_Num+1):
            p = np.exp((-1/2)*np.dot(np.dot(np.transpose(Test - means[:, j-1]),np.linalg.inv(covmat)),(Test - means[:, j-1])));
            if p>p_max:
                class_name = j
                p_max = p
        if class_name == ytest[i-1]:
            match = match + 1
        ypreds.append(class_name)
    acc=(match/float(Data_Num))*100
    ypred = np.transpose(ypreds)
    print(acc)
    return acc,ypred



def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    Data_Num = np.shape(Xtest)[0]
    Class_Num = np.shape(means)[1]
    ypreds=[]
    match = 0
    for i in range(1,Data_Num+1):
        p_max = -10000
        class_name = 0
        Test = np.transpose(Xtest[i-1,:])
        for j in range(1, Class_Num+1):
            FirstItem = -(np.linalg.inv(covmats[j-1]))/2
            SecondItem = np.dot(np.linalg.inv(covmats[j-1]),means[:,j-1])
            ThirdItem = np.dot(np.dot(-(np.transpose(means[:,j-1]))/2,np.linalg.inv(covmats[j-1])),means[:,j-1])
            FourthItem = -np.log(np.linalg.det(covmats[j-1]))/2
            p = np.dot(np.transpose(Test),np.dot(FirstItem,Test))+np.dot(np.transpose(SecondItem),Test)+ThirdItem+FourthItem
            if p>p_max:
                class_name = j
                p_max = p
        if class_name == ytest[i-1]:
            match = match + 1
        ypreds.append(class_name)
    acc=(match/float(Data_Num))*100
    ypred = np.transpose(ypreds)
    print(acc)
    return acc,ypred


def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    xTran = np.transpose(X)
    mulres = np.matmul(xTran, X)
    XTranXInv = inv(mulres)
    XTranY = np.matmul(xTran, y)
    w = np.matmul(XTranXInv, XTranY)

    # IMPLEMENT THIS METHOD
    return w


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    xTran = np.transpose(X)
    I = np.eye(len(X[0]))
    for i in range(len(I)):
        I[i][i] = lambd
    XTranXInv = inv(np.dot(xTran, X) + I)
    XTranY = np.dot(xTran, y)
    w = np.dot(XTranXInv, XTranY)

    # IMPLEMENT THIS METHOD
    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    pred = np.matmul(Xtest, w)
    diff = np.subtract(ytest, pred)
    prod = np.matmul(np.transpose(diff), diff)
    mse = prod[0][0] / len(ytest)

    # IMPLEMENT THIS METHOD
    return mse


def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    pred = np.transpose(np.matrix(np.dot(X, w)))
    diff = np.subtract(y, pred)
    prod = 0.5 * np.dot(np.transpose(diff), diff)
    wTw = np.dot(np.transpose(w), w)
    error = prod + (0.5 * np.dot(lambd, wTw))
    error_grad = np.transpose(np.matrix(np.dot(lambd, w))) - np.dot(np.transpose(X), diff)
    return np.squeeze(np.asarray(error)), np.squeeze(np.asarray(error_grad))


def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))

    n = len(x)
    Xp = np.ones((n, p+1))
    for i in range(1, p+1):
        Xp[:, i] = x**i

    # IMPLEMENT THIS METHOD
    return Xp


# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
# means, covmat = ldaLearn(X, y)
# ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
# print('LDA Accuracy = ' + str(ldaacc))
# # QDA
# means, covmats = qdaLearn(X, y)
# qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
# print('QDA Accuracy = ' + str(qdaacc))
#
# # plotting boundaries
# x1 = np.linspace(-5, 20, 100)
# x2 = np.linspace(-5, 20, 100)
# xx1, xx2 = np.meshgrid(x1, x2)
# xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
# xx[:, 0] = xx1.ravel()
# xx[:, 1] = xx2.ravel()
#
# fig = plt.figure(figsize=[12, 6])
# plt.subplot(1, 2, 1)
#
# zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
# plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
# plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
# plt.title('LDA')
#
# plt.subplot(1, 2, 2)
#
# zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
# plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
# plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
# plt.title('QDA')
#
# plt.show()
# Problem 2
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)
mle_train = testOLERegression(w, X, y)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)
mle_train_i = testOLERegression(w_i, X_i, y)

print("---------Linear Regression---------")
print('MSE without intercept for testing data: ' + str(mle))
print('MSE without intercept for training data: ' + str(mle_train))

print('MSE with intercept for testing data: ' + str(mle_i))
print('MSE with intercept for training data: ' + str(mle_train_i))
print()

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)

mse_test_min = np.min(mses3)
mse_train_min = np.min(mses3_train)
opt_lambda = 0
opt_lambda_index = 0
for i in range(0, lambdas.shape[0]):
    if mses3[i] == mse_test_min:
        opt_lambda = lambdas[i]
        opt_lambda_index = i
        break

# color = np.where(mask, 'red', 'blue')
print("---------Ridge Regression---------")
print('Minimum value of MSE for testing data with intercept: ', mse_test_min)
print('Value of MSE for training data with intercept at the optimal testing data lambda value: ',
      mses3_train[opt_lambda_index][0])
print('Optimal lambda is: ', opt_lambda)
# plt.plot(x[1:], y[1:], 'ro')

plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.plot(opt_lambda, mses3_train[opt_lambda_index], 'rx', )
plt.xticks([opt_lambda, 0.2, 0.4, 0.6, 0.8])
plt.yticks([2200, 2400, int(mses3_train[opt_lambda_index][0]), 2600, 2800, 3000, 3200])
plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3)
plt.xticks([opt_lambda, 0.2, 0.4, 0.6, 0.8])
plt.yticks([int(mse_test_min), 3000, 3200, 3400, 3600, 3800])
plt.plot(opt_lambda, mse_test_min, 'rx', )
plt.title('MSE for Test Data')

plt.show()

weight_diff = np.subtract(np.abs(w_l), np.abs(w_i))

for weight in weight_diff:
    

print('difference between ridge regression weights and linear regression weights: ', weight_diff)
# # Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 20}  # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = opt_lambda  # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax, 2))
mses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    mses5_train[p, 0] = testOLERegression(w_d1, Xd, y)
    mses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    mses5_train[p, 1] = testOLERegression(w_d2, Xd, y)
    mses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization', 'Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization', 'Regularization'))
plt.show()

