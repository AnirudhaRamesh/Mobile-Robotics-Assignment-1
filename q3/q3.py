import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl



if __name__ == '__main__':
	#Define world-co-ords
	#Define image co-ords
	t1 = np.matrix([[284.56243896, 149.29254150],
		[373.93179321, 128.26719666],
		[387.53588867, 220.22708130],
		[281.29962158, 241.72782898]])

	t2 = np.matrix([[428.86453247, 114.50731659],
		[524.76373291, 92.092185970],
		[568.36596680, 180.55757141],
		[453.60995483, 205.22370911]])

	c = np.float(0.0790)

	world_co_ords1 = np.matrix([[0,0.1315],[0.1315,0.1315],[0.1315,0.0],[0,0]])
	world_co_ords2 = np.matrix([[0.1315+c,0.1315],
		[0.1315+c+0.1315,0.1315],
		[0.1315+c+0.1315,0],[0.1315+c,0]])

	M = np.zeros((16,9), dtype=np.float64)


	for i in range(4):
		X = world_co_ords1[i,0]
		Y = world_co_ords1[i,1]
		x = t1[i,0]
		y = t1[i,1]
		M[2*i]     = [-X,-Y,-1,  0,  0, 0, x*X ,x*Y ,x]
		M[2*i + 1] = [ 0, 0, 0, -X, -Y,-1, y*X ,y*Y ,y]

	for i in range(4):
		X = world_co_ords2[i,0]
		Y = world_co_ords2[i,1]
		x = t2[i,0]
		y = t2[i,1]

		M[2*i + 8]     = [-X,-Y,-1,  0,  0, 0, x*X ,x*Y ,x]
		M[2*i + 9]     = [ 0, 0, 0, -X, -Y,-1, y*X ,y*Y ,y]

	u,s,vh = np.linalg.svd(M)
	vecHt = vh[8,:]
	H = np.reshape(vecHt,(3,3))

	z = np.ones((4,1), dtype=np.int64)
	wc1_homo = np.append(world_co_ords1, z, axis = 1)
	wc2_homo = np.append(world_co_ords2, z, axis = 1)
	# wc1.T

	im = plt.imread("image.png")
	plt.imshow(im)
	# plt.gca().invert_yaxis()

	im_coord1 = H@wc1_homo.T
	im_coord2 = H@wc2_homo.T

	im_coord1 = im_coord1.T
	im_coord2 = im_coord2.T


	im_coord1[:,0] /= im_coord1[:,2]
	im_coord1[:,1] /= im_coord1[:,2]
	im_coord1[:,2] /= im_coord1[:,2]

	im_coord2[:,0] /= im_coord2[:,2]
	im_coord2[:,1] /= im_coord2[:,2]
	im_coord2[:,2] /= im_coord2[:,2]


	plt.scatter([im_coord1[:,0]],[im_coord1[:,1]], s=60,marker='s',edgecolors='none')
	plt.scatter([im_coord2[:,0]],[im_coord2[:,1]], s=60,marker='s',edgecolors='none',color='r')

	# plt.scatter(im_coords[:,0],im_coords[:,1], s=5, c = points[:,0], cmap=mpl.cm.get_cmap('nipy_spectral'))
	plt.scatter([t1[:,0].T],[t1[:,1].T], s=120, marker='*',edgecolors='none')
	plt.scatter([t2[:,0].T],[t2[:,1].T], s=120, marker='*',edgecolors='none')



	"""
	big break
	"""




	#Decomposing H
	K = np.matrix([[406.952636, 0.000000, 366.184147],
		[0.000000, 405.671292, 244.705127],
		[0.000000, 0.000000, 1.000000]])

	K_inv = np.linalg.inv(K)

	MatH = K_inv@H
	h1 = MatH[:,0]
	h2 = MatH[:,1]
	h3 = MatH[:,2]

	t  = MatH[:,2]/np.linalg.norm(h1)

	temp_R = np.zeros((3,3), dtype=np.float64)
	temp_R[0] = h1.T
	temp_R[1] = h2.T
	temp_R[2] = np.cross(h2.T, h1.T)

	temp_R = temp_R.T

	utemp_R, stemp_R, vhtemp_R = np.linalg.svd(temp_R)

	R = utemp_R@vhtemp_R
	#
	temp = np.matrix(t)
	R[:,2] = temp.T

	im_coord1 = K@R@wc1_homo.T 
	im_coord2 = K@R@wc2_homo.T 

	im_coord1 = im_coord1.T
	im_coord2 = im_coord2.T


	im_coord1[:,0] /= im_coord1[:,2]
	im_coord1[:,1] /= im_coord1[:,2]
	im_coord1[:,2] /= im_coord1[:,2]

	im_coord2[:,0] /= im_coord2[:,2]
	im_coord2[:,1] /= im_coord2[:,2]
	im_coord2[:,2] /= im_coord2[:,2]


	#plt.scatter([im_coord1[:,0]],[im_coord1[:,1]], s=120,marker='x',edgecolors='none', color ='r')
	#plt.scatter([im_coord2[:,0]],[im_coord2[:,1]], s=120,marker='x',edgecolors='none', color ='r')
	plt.show()
