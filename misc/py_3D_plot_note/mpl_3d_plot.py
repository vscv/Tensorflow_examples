from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
#if using a Jupyter notebook, include:
#%matplotlib inline


#x = np.arange(-5,5,.5)
#y = np.arange(-5,5,.5)
#X,Y = np.meshgrid(x,y)
#Z = X*np.exp(-X**2 - Y**2)
#
#
#fig = plt.figure(figsize=(6,6))
#ax = fig.add_subplot(111, projection='3d')


# Plot a 3D surface
#ax.plot_surface(X, Y, Z)
##ax.plot_surface(X, Y, Z, rstride=1, cstride=1,shade=True, alpha=0.3)

# Plot a basic wireframe.
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

#plt.show()

##############
##############

"""
FV_value: set 0268212.txt to 8x8, pad with '0'.

0268212.txt
11.88568,11.062547,11.9634695,11.085825,
14.704241,15.178696,12.561343,14.907657,15.055266,11.638041,
15.772865,19.790277,16.376902,18.42167,17.43712,16.798529,13.210518,10.31165,
19.347595,11.789717,18.261658,23.76924,20.169788,16.463709,16.25219,11.579194,
21.83724,1.2624167,21.867619,25.419193,20.919302,18.244373,18.87963,14.420412,
21.268051,19.675962,20.257395,20.420235,20.500137,18.389208,20.531063,15.618982,
18.646898,21.697271,17.999739,20.868557,19.391293,15.656485,
17.530151,19.134745,17.971888,16.919024,
"""

x = np.arange(0,8,1)
y = np.arange(0,8,1)
X,Y = np.meshgrid(x,y)
#Z = X*np.exp(-X**2 - Y**2)
Z = np.array([
[0,0,11.88568,11.062547,11.9634695,11.085825,0,0],
[0,14.704241,15.178696,12.561343,14.907657,15.055266,11.638041,0],
[15.772865,19.790277,16.376902,18.42167,17.43712,16.798529,13.210518,10.31165],
[19.347595,11.789717,18.261658,23.76924,20.169788,16.463709,16.25219,11.579194],
[21.83724,1.2624167,21.867619,25.419193,20.919302,18.244373,18.87963,14.420412],
[21.268051,19.675962,20.257395,20.420235,20.500137,18.389208,20.531063,15.618982],
[0,18.646898,21.697271,17.999739,20.868557,19.391293,15.656485,0],
[0,0,17.530151,19.134745,17.971888,16.919024,0,0]
])

Z_2 = Z + np.random.rand(8,8) * 10

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D surface
#ax.plot_surface(X, Y, Z)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet',shade=True, alpha=1) # plot_surface no "label"

ax.plot_wireframe(X, Y, Z_2, rstride=1, cstride=1, color='m', alpha=0.3, label="ground truth")


# Plot a basic wireframe.
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

plt.legend()
plt.show()
