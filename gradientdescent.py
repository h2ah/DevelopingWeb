import numpy as np
############################
# Problem Model

points = [(np.array([2]),4),(np.array([4]),2)]
d = 1

def F(w):
    return sum((w.dot(x) - y)**2 for x,y in points)

def dF(w):
    return sum(2*(w.dot(x) - y) * x for x,y in points)

#############################
# Algorithm: Problem Solution# Gradient descent

def gradientDescent(F, Df, d):
    # Gradient descent
    eta = 0.01
    w = np.zeros(d)
    for t in range(100):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t,w,value))

gradientDescent(F,dF,d)