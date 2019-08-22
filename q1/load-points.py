import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def load_velodyne_points(points_path):
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:,:3]                # exclude reflectance values, becomes [X Y Z]
    points = points[1::5,:]              # remove every 5th point for display speed (optional)
    points = points[(points[:,0] > 5)]   # remove all points behind image plane (approximate)
    return points

def R_mat(theta, axis):
	c = np.cos(theta)
	s = np.sin(theta)
	if axis == "Z":
		return np.matrix([[c,s,0],[-s,c,0],[0,0,1]])
	if axis == "Y":
		return np.matrix([[c,0,-s],[0,1,0],[s,0,c]])
	if axis == "X":
		return np.matrix([[1,0,0],[0,c,s],[0,-s,c]])

if __name__ == '__main__':
    points = load_velodyne_points('lidar-points.bin')
    #convert points to homogeneous co-ordinates 
    z = np.ones((6965,1), dtype=np.int64)
    points_homo = np.append(points,z, axis = 1)
    points_homo = np.transpose(points_homo)
    I_C = np.matrix([[1,0,0,-0.27],[0,1,0,-0.06],[0,0,1,0.08]])
    R = R_mat(-np.pi/2, "X")@R_mat(-np.pi/2, "Z")
    K = np.matrix([[7.215377e+02, 0.000000e+00, 6.095593e+02],[0.000000e+00, 7.215377e+02, 1.728540e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    im_coords = K@R@I_C@points_homo
    im_coords = im_coords.T
    im_coords[:,0] /= im_coords[:,2]
    im_coords[:,1] /= im_coords[:,2]
    im_coords[:,2] /= im_coords[:,2]
    print(im_coords)
    # final_im = im_coords[0,:]/im_coords[2,:]
    # p_c = np.matrix(im_coords[0,:]/im_coords[2,:],im_coords[1,:]/im_coords[2,:])

    im = plt.imread("image.png")
    plt.imshow(im)
    # plt.gca().invert_yaxis()

    # plt.scatter(im_coords[:,0],im_coords[:,1], s=5, c = points[:,0], cmap=mpl.cm.get_cmap('nipy_spectral'))
    plt.scatter([im_coords[:,0].T],[im_coords[:,1].T], s=60,c=[points[:,0].T],cmap=mpl.cm.get_cmap('nipy_spectral'),marker='*',edgecolors='none')
    plt.show()