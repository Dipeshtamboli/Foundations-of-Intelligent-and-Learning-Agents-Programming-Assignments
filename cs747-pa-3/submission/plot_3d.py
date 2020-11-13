import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
zs = np.load("al_epsi.npy")
print(zs.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.arange(0,1,0.1)
y = np.arange(0,1,0.1)

xs, ys = np.meshgrid(x, y)


ax.plot_wireframe(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
ax.set_xlabel('Epsilon')
ax.set_ylabel('Alpha')
ax.set_zlabel('Episodes reached')
ax.set_title("Episodes reached in 10000 steps for varying alpha and epsilon")
elev = 0
azim = -90
ax.view_init(elev=elev, azim=azim)
plt.savefig(f"3d_{len(xs)*len(ys)}points_elev:{elev}_azim:{azim}.jpg")
# plt.show()