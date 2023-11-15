from sympy.physics.hydrogen import Psi_nlm
from sympy import Symbol, conjugate, pi, Abs, integrate
import numpy as np
from sympy.utilities.lambdify import lambdify
from sympy.abc import n, l, r, phi, theta
import open3d as o3d
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import seaborn as sns

def create_point_cloud(length, interval):
    #Create a 3D grid of points
    step = 1/interval
    x = np.arange(-length, length, step)
    y = np.arange(-length, length, step)
    z = np.arange(-length, length, step)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z) 

    grid_r = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
    grid_theta = np.arccos(grid_z/grid_r)
    grid_phi = np.sign(grid_y) * np.arccos(grid_x/np.sqrt(grid_x**2 + grid_y**2))

    grid_xyz = [grid_x, grid_y, grid_z]
    grid_rtp = [grid_r, grid_theta, grid_phi]

    return grid_xyz, grid_rtp

def get_wavefunction_grid(grid_rtp):
    grid_r, grid_theta, grid_phi = grid_rtp
    grid_amplitude = np.vectorize(get_wavefunction_amplitude)(grid_r, grid_theta, grid_phi)

    return grid_amplitude

def get_color_grid(grid_amplitude):
    amplitude = np.abs(grid_amplitude)
    real_cgrid = grid_amplitude.real/amplitude

    amplitude[np.isnan(amplitude)] = 0
    amplitude[np.isinf(amplitude)] = 0

    real_cgrid[np.isnan(real_cgrid)] = 0
    real_cgrid[np.isinf(real_cgrid)] = 0

    return real_cgrid, amplitude

get_wavefunction_amplitude =lambdify((r, phi, theta), Psi_nlm(4,1,1, r, phi, theta, 1), ('numpy', 'math', 'sympy'))

if __name__ == "__main__":
    grid_xyz, grid_rtp = create_point_cloud(35,2)

    func_amplitude = get_wavefunction_grid(grid_rtp)

    real_amplitude, amplitude = get_color_grid(func_amplitude)

    #renormalize opacity_grid
    rescaled_real_amplitude = real_amplitude/np.max(real_amplitude)
    real_amplitude_arr = rescaled_real_amplitude.flatten()

    rescaled_amplitude = amplitude/np.max(amplitude)
    alpha_plot = rescaled_amplitude.flatten()

    cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    colors_plot = cmap(real_amplitude_arr)
    # cmap = plt.get_cmap('RdYlBu')
    # colors_plot = cmap(real_amplitude_arr)

    #make x,y,z points into Nx3 array
    points = np.array([grid_xyz[0].flatten(), grid_xyz[1].flatten(), grid_xyz[2].flatten()]).T

    #Only display points with non-zero amplitude
    #get median of amplitude
    q_cutoff = np.quantile(alpha_plot, 0.75)

    points = points[alpha_plot > q_cutoff]
    colors_plot = colors_plot[alpha_plot > q_cutoff]
    alpha_plot = alpha_plot[alpha_plot > q_cutoff]

    #create array where each line is x y z r g b a 
    #where x y z are the coordinates of the point
    #and r g b a are the color values
    #in this format np.array([(0, 0, 0),(0, 1, 1),(1, 0, 1),(1, 1, 0)],dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    points_to_ply = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1'),('alpha', 'u1')])
    points_to_ply['x'] = points[:,0]
    points_to_ply['y'] = points[:,1]
    points_to_ply['z'] = points[:,2]
    points_to_ply['red'] = (colors_plot[:,0]*255).astype('u1')
    points_to_ply['green'] = (colors_plot[:,1]*255).astype('u1')
    points_to_ply['blue'] = (colors_plot[:,2]*255).astype('u1')
    points_to_ply['alpha'] = (alpha_plot*255).astype('u1')


    el = PlyElement.describe(points_to_ply, 'vertex')
    PlyData([el], text=True).write('data/test3_irm4.ply')
