# -*- coding: utf-8 -*-
"""Untitled29.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BrIM16TdX4rA8PsnOWhEtNI-2tFjBxwk
"""

import numpy as np
def force_vec(s,t,n):
  z = np.column_stack([s,t,n])
  return z

s1 = np.array([0, 1, 0])
t1 = np.array([0, 0, 1])
n1 = np.array([1, 0, 0])
z1=  force_vec(s1,t1,n1)


s2 = np.array([0, 1, 0])
t2 = np.array([0, 0, -1])
n2 = np.array([-1, 0, 0])
z2=  force_vec(s2,t2,n2)

s3 = np.array([0, 1, 0])
t3 = np.array([0, 0, -1])
n3 = np.array([-1, 0, 0])
z3 = force_vec(s3,t3,n3)


#define function to calculate cross product
def cross_prod(x,s):
    A = [x[1]*s[2] - x[2]*x[1],
            x[2]*s[0] - x[0]*s[2],
            x[0]*s[1] - x[1]*s[0]]
    p = np.transpose(A)
    return p
def cross_prod(x,t):
    A = [x[1]*t[2] - x[2]*t[1],
            x[2]*t[0] - x[0]*t[2],
            x[0]*t[1] - x[1]*t[0]]
    q = np.transpose(A)
    return q
def cross_prod(x,n):
    A = [x[1]*n[2] - x[2]*n[1],
            x[2]*n[0] - x[0]*n[2],
            x[0]*n[1] - x[1]*n[0]]
    r = np.transpose(A)
    return r

# test
s1 = np.array([0, 1, 0])
t1 = np.array([0, 0, 1])
n1 = np.array([1, 0, 0])
x1 = np.array([-1, -0.5, 0])
p1 = cross_prod(x1,s1)
q1 = cross_prod(x1,t1)
r1 = cross_prod(x1,n1)
k= np.column_stack([p1,q1,r1])
G1 = np.concatenate([z1,k])

s2 = np.array([0, 1, 0])
t2 = np.array([0, 0, -1])
n2 = np.array([-1, 0, 0])
x2 = np.array([-1, 0.5, 0])
p2 = cross_prod(x2,s2)
q2 = cross_prod(x2,t2)
r2 = cross_prod(x2,n2)
l= np.column_stack([p2,q2,r2])
G2 = np.concatenate([z2,l])

s3 = np.array([0, 1, 0])
t3 = np.array([0, 0, -1])
n3 = np.array([-1, 0, 0])
x3 = np.array([1, 0, 0])
p3 = cross_prod(x3,s3)
q3 = cross_prod(x3,t3)
r3 = cross_prod(x3,n3)
m= np.column_stack([p3,q3,r3])

G3 = np.concatenate([z3,m])
G= np.block([G1,G2,G3])
print(G)

