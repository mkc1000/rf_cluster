import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotation_matrix(angle,axis):
    rot2d = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    twobythree = np.insert(rot2d,axis,np.array([0,0]),axis=0)
    output = np.insert(twobythree,axis,np.array([0,0,0]),axis=1)
    output[axis][axis] = 1
    return output

def rotate_cloud(cloud, angle, axis):
    rot_mat = rotation_matrix(angle, axis)
    return (rot_mat.dot(cloud.T)).T

def impossible_triangle(n):
    sphere1 = np.random.normal(size=(n/3,3))
    sphere2 = np.random.normal(size=(n/3,3))
    sphere3 = np.random.normal(size=(n/3,3))

    sphere1[:,0] = sphere1[:,0]*10
    sphere1 = rotate_cloud(sphere1, np.pi/12, 1)

    sphere2[:,0] = sphere2[:,0]*10
    sphere2 = rotate_cloud(sphere2, np.pi/12, 1)

    sphere3[:,0] = sphere3[:,0]*10
    sphere3 = rotate_cloud(sphere3, np.pi/12, 1)

    sphere2 = rotate_cloud(sphere2, 2*np.pi/3, 2)
    sphere3 = rotate_cloud(sphere3, -2*np.pi/3, 2)

    sphere1 = sphere1 + np.array([[0,-10,0]])
    sphere2 = sphere2 + np.array([[6,10,0]])
    sphere3 = sphere3 + np.array([[-6,10,0]])

    return np.vstack((sphere1,sphere2,sphere3))

def skewer(n):
    sphere1 = np.random.normal(size=(n/3,3))
    sphere2 = np.random.normal(size=(n/3,3))
    sphere3 = np.random.normal(size=(n/3,3))

    sphere1[:,2] = sphere1[:,2]*10
    sphere2[:,0] = sphere2[:,0]*10
    sphere2[:,1] = sphere2[:,1]*3
    sphere2[:,2] = sphere2[:,2]/10
    sphere3[:,0] = sphere3[:,0]*10
    sphere3[:,1] = sphere3[:,1]*3
    sphere3[:,2] = sphere3[:,2]/10
    sphere3 = rotate_cloud(sphere3, np.pi/2, 2)
    sphere1 = sphere1 + np.array([0,0,1])
    sphere2 = sphere2 + np.array([0,0,2])
    sphere3 = sphere3 - np.array([0,0,2])

    return np.vstack((sphere1,sphere2,sphere3))


def plot_3d(arr, color='b'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.scatter(ax,arr[:,0],arr[:,1],arr[:,2],color=color)
    plt.show()

def color_by_assignment(assignments):
    colors = ['r','b','g','y','m','c','k']
    color_dict = {val: colors[i] for i, val in enumerate(np.unique(np.array(assignments)))}
    colors = [color_dict[ass] for ass in assignments]
    return colors
