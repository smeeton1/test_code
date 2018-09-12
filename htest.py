import sys
import numpy 
import cmath 
import math 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate

xrang= numpy.arange(-2,2,0.01)
yrang= numpy.arange(-2,2,0.01)


for p in range(0,5)
  VeB=[0,0,0,0,0,0,0,0,0,0,0]
  VeB[p]=1

  rho=numpy.outer(VeB,VeB)
  nrho=len(rho)

  Qfun=[[0 for i in xrange(len(xrang))] for i in xrange(len(yrang))]

  h=0
  g=0
  alpha=numpy.empty([nrho],dtype=complex)
  for i in xrang:
    for l in yrang:
      for k in range(0,nrho):
	alpha[k]=cmath.exp(-pow(abs(i+1j*l),2)/2)*(pow((i+1j*l),k)/math.sqrt(math.factorial(k)))
	
      Qfun[g][h]=(numpy.dot(numpy.transpose(alpha),numpy.dot(rho,alpha.conjugate()))).real
      g=g+1
      
    h=h+1
    g=0
    

  X, Y = numpy.meshgrid(xrang, yrang)

  QfunName='Qfun1_%i.eps' % (p)
  fig = plt.figure(3)#,figsize=(3,2))
  #ax = fig.add_subplot(111, projection='3d')
  #ax.plot_surface(X,Y,Qfun)
  plt.matshow(Qfun)
  plt.savefig(QfunName)

  QfunName3d='Qfun3D1_%i.eps' % (p)
  fig = plt.figure(4)#,figsize=(3,2))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X,Y,Qfun)
  plt.savefig(QfunName3d)