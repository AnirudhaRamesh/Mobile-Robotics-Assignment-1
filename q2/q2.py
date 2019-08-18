import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np
from math import cos, sin, pi 
import matplotlib.patches as patches

img = plt.imread('image.png')

K = np.array([[7.2153e+02,0,6.0955e+02],[0,7.2153e+02,1.7285e+02],[0,0,1]])
camera_height = 1.65 
car_height = 1.38 
car_width = 1.51 
car_length = 4.10 

# Car coordinates from image (Pixel Coordinates)

# Bottom near right 
x3 = 953
y3 = 300 
# Front Left bottom 
x4 = 745 
y4 = 256

n = [0,-1,0]

b = np.array([[(x3+x4)/2],[y3] , [1]])

B_num = -camera_height * np.matmul(np.linalg.inv(K),b)

B_den = np.matmul(n,np.matmul(np.linalg.inv(K),b))

B = B_num / B_den

fig, ax = plt.subplots(1)
ax.imshow(img)

B_centered = B + [[0],[-car_height/2] , [car_length]]

pts_list_2d = [[i+j for i in range(2)] for j in range(8)]
pts_list_3d = [[i+j for i in range(2)] for j in range(8)]

pts_list_3d[0] = B_centered + [[(car_width+1)/2],[(car_height+0.5)/2] , [(car_length+1.25)/2]]
pts_list_3d[1] = B_centered + [[(car_width+0.75)/2],[(car_height+0.5)/2] , [-(car_length+0.75)/2]]
pts_list_3d[2] = B_centered + [[(car_width+1)/2],[-(car_height+0.5)/2] , [(car_length+1.25)/2]]
pts_list_3d[3] = B_centered + [[(car_width+0.75)/2],[-(car_height+0.5)/2] , [-(car_length+0.75)/2]]
pts_list_3d[4] = B_centered + [[-(car_width+1)/2],[(car_height+0.5)/2] , [(car_length+1.25)/2]]
pts_list_3d[5] = B_centered + [[-(car_width+0.75)/2],[(car_height+0.5)/2] , [-(car_length+0.75)/2]]
pts_list_3d[6] = B_centered + [[-(car_width+1)/2],[-(car_height+0.5)/2] , [(car_length+1.25)/2]]
pts_list_3d[7] = B_centered + [[-(car_width+0.75)/2],[-(car_height+0.5)/2] , [-(car_length+0.75)/2]]

ang  = 5 * pi/180 
rot_y = [[cos(ang), 0, sin(ang)],[0,1,0],[-sin(ang), 0, cos(ang)]]

for i in range(8) : 
    pts_list_3d[i] = np.matmul(rot_y, pts_list_3d[i])
    pos_3d = np.matmul(K,pts_list_3d[i])
    pts_list_2d[i] = pos_3d[0:2]/pos_3d[2:3] 
    plt.scatter(pts_list_2d[i][0], pts_list_2d[i][1])

x = [i[0] for i in pts_list_2d]
y = [i[1] for i in pts_list_2d]
# plt.plot(x,y,'-o')

rect = patches.Rectangle((x[5],y[5]),x[3]-x[5],y[7]-y[5],linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)
rect = patches.Rectangle((x[4],y[4]),x[2]-x[4],y[6]-y[4],linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)

right_rect_x = [i for i in x[0:4]]
right_rect_y = [i for i in y[0:4]]
plt.plot(right_rect_x,right_rect_y,'-r')

left_rect_x = [i for i in x[4:8]]
left_rect_y = [i for i in y[4:8]]
plt.plot(left_rect_x,left_rect_y,'-r')


plt.show()
