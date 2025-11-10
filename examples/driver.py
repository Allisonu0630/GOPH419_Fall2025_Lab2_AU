import numpy as np
import os
import matplotlib.pyplot as plt
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.lab_02.linalg_interp import spline_function

def load_density_data(filepath):
    """
    Load temperature and density data from a two-column text file.
    Returns:
        temp: ndarray of temperatures
        density: ndarray of densities
    """
    data = np.loadtxt(filepath)
    temp = data[:, 0]
    density = data[:, 1]
    return temp, density

def evaluate_spline(temp, density, order, label):
    """
    Fit spline of given order and evaluate on a dense temperature grid.
    Returns:
        temp_dense: evaluation points
        density_interp: interpolated values
    """
    spline = spline_function(temp, density, order=order)
    temp_dense = np.linspace(temp[0], temp[-1], 500)
    density_interp = spline(temp_dense)
    return temp_dense, density_interp

def plot_density(temp, density, temp_dense, density_interp, title, ylabel, filename):
    """
    Plot and save original data and interpolated spline to examples/driver.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(temp, density, 'o', label='Data')
    plt.plot(temp_dense, density_interp, '-', label='Spline')
    plt.title(title)
    plt.xlabel('Temperature (°C)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_dir = os.path.dirname(__file__)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()

def run_density_interpolation():
    # Paths to data files
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', 'data')
    water_file = os.path.join(data_dir, 'water_density_vs_temp_usgs.txt')
    air_file = os.path.join(data_dir, 'air_density_vs_temp_eng_toolbox.txt')

    # Load data
    temp_water, rho_water = load_density_data(water_file)
    temp_air, rho_air = load_density_data(air_file)

    # Evaluate and plot water density spline
    for order in [1, 2, 3]:
        temp_dense, rho_interp = evaluate_spline(temp_water, rho_water, order, label='Water')
        plot_density(temp_water, rho_water, temp_dense, rho_interp,
                     title=f'Water Density Spline (Order {order})',
                     ylabel='Density (g/cm³)', filename=f'water_spline_order_{order}.png')

    # Evaluate and plot air density spline
    for order in [1, 2, 3]:
        temp_dense, rho_interp = evaluate_spline(temp_air, rho_air, order, label='Air')
        plot_density(temp_air, rho_air, temp_dense, rho_interp,
                     title=f'Air Density Spline (Order {order})',
                     ylabel='Density (kg/m³)', filename=f'air_spline_order_{order}.png')

if __name__ == "__main__":
    run_density_interpolation()