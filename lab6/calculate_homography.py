import numpy as np

# original rectangle corners
src = np.array([
    [0,0],
    [0,3],
    [5,3],
    [5,0]
])

# transformed corners
dst = np.array([
    [1,1],
    [3,3],
    [6,3],
    [5,2]
])

A = []
# DLT Algorithm
for (x,y), (xp,yp) in zip(src,dst):
    A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
    A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])

A = np.array(A)

# solve Ah = 0 using SVD
U, S, Vt = np.linalg.svd(A)

H = Vt[-1].reshape(3,3)

# normalize
H = H / H[-1,-1]

print("Homography matrix:")
print(H)