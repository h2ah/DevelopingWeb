import numpy as np
############################
# Problem Model

#points = [(np.array([2]),4),(np.array([4]),2)]
#d = 1

# Generate artifical data

true_w = np.array([1,2,3,4,5])
d = len(true_w)
points = []
for i in range(1000):
    x = np.random.rand(d)
    y = true_w.dot(x) + np.random.rand()
    # prsint(x,y)
    points.append((x,y))

def F(w):
    return sum((w.dot(x) - y)**2 for x,y in points) / len(points)

def dF(w):
    return sum(2*(w.dot(x) - y) * x for x,y in points) / len(points)

def sF(w,i):
    x, y = points[i]
    return ((w.dot(x) - y)**2)

def sdF(w,i):
    x, y = points[i]
    return 2*(w.dot(x) - y) * x
#############################
# Algorithm: Problem Solution# Gradient descent

def gradientDescent(F, Df, d):
    # Gradient descent
    eta = 0.01
    w = np.zeros(d)
    for t in range(1000):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t,w,value))

def stochasticGradientDescent(sF, sDf, d, n):
    # Stochastic Gradient descent
    eta = 0.01
    w = np.zeros(d)
    for t in range(1000):
        numUpdates = 0
        for i in range(100):
            value = sF(w,i)
            numUpdates += 1
            gradient = sdF(w,i)
            eta = 1 / numUpdates
            w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t,w,value))

stochasticGradientDescent(sF,sdF,d,len(points))