
"""
class VelocityField
"""

import sys
import numpy as np
import netCDF4


class VelocityField:
    """
    Data file

    Loads the input file with the NetCFD (.nc) 
    format or a Tecplot format (.dat); 
    initialize the variables.

    :param file_path: file path
    :type  file_path: str
    :param time_step: current time step
    :type  time_step: int
    :param mean_file_path: in case of a mean field subtraction
    :type  mean_file_path: str
    :param file_type: 'piv_netcdf', 'dns, 'dns2', 'piv_tecplot', 'openfoam'
    :type  file_type: str
    :param x_coordinate_matrix: spatial mesh
    :type  x_coordinate_matrix: ndarray
    :param y_coordinate_matrix: spatial mesh
    :type  y_coordinate_matrix: ndarray
    :param z_coordinate_matrix: spatial mesh, optional
    :type  z_coordinate_matrix: ndarray
    :param u_velocity_matrix: 1st component velocity field
    :type  u_velocity_matrix: ndarray
    :param v_velocity_matrix: 2nd component velocity field
    :type  v_velocity_matrix: ndarray
    :param w_velocity_matrix: 3rd component velocity field, optional
    :type  w_velocity_matrix: ndarray
    :param normalization_flag: for normalization of the swirling field
    :type  normalization_flag: boolean
    :param normalization_direction: 'None', 'x' or 'y'
    :type  normalization_direction: str
    :param x_coordinate_step: for homogeneous mesh, provides a unique step
    :type  x_coordinate_step: float
    :param y_coordinate_step: for homogeneous mesh, provides a unique step 
    :type  y_coordinate_step: float
    :param z_coordinate_step: for homogeneous mesh, provides a unique step 
    :type  z_coordinate_step: float
    :param derivative: contains 'dudx', 'dudy', 'dvdx', 'dvdy'. 
                       Can be extended to the 3rd dimension
    :type  derivative: dict
    :returns: vfield, an instance of the VelocityField class
    :rtype: class VelocityField
    """

    def __init__(self, file_path="/", time_step=0, mean_file_path="/", file_type="/"):

        if '{:' in file_path:
            self.file_path = file_path.format(time_step)
        else:
            self.file_path = file_path

        self.time_step = time_step
        self.mean_file_path = mean_file_path

        if file_type == 'Reconstructed_2D_data':
            try:
                datafile_read = np.loadtxt(self.file_path, delimiter=" ", dtype=np.float32)
            except IOError:
                sys.exit("\nReading error. Maybe a wrong file type?\n")

            index_x, index_y, index_u, index_v, index_d, index_M = 0, 1, 2, 3, 4, 5
            dx_tmp = np.array(datafile_read[:, index_x])

            for i in range(1, dx_tmp.shape[0]):
                if dx_tmp[i] == dx_tmp[0]:
                    self.y_coordinate_size = i
                    break
            self.x_coordinate_size = np.int(dx_tmp.shape[0] / self.y_coordinate_size)  # domain size

            self.u_velocity_matrix = np.array(datafile_read[:, index_u]).reshape(self.x_coordinate_size,
                                                                                 self.y_coordinate_size)
            self.v_velocity_matrix = np.array(datafile_read[:, index_v]).reshape(self.x_coordinate_size,
                                                                                 self.y_coordinate_size)

            if self.mean_file_path != '/':
                print("subtracting mean file")
                # load and subtract mean data
                datafile_mean_read = np.loadtxt(mean_file_path, delimiter=" ", dtype=np.float32)
                u_velocity_matrix_mean = np.array(datafile_mean_read[:, index_u]).reshape(self.x_coordinate_size,
                                                                                          self.y_coordinate_size)
                v_velocity_matrix_mean = np.array(datafile_mean_read[:, index_v]).reshape(self.x_coordinate_size,
                                                                                          self.y_coordinate_size)
                self.u_velocity_matrix = self.u_velocity_matrix - u_velocity_matrix_mean
                self.v_velocity_matrix = self.v_velocity_matrix - v_velocity_matrix_mean

            tmp_x = np.array(datafile_read[:, index_x]).reshape(self.x_coordinate_size, self.y_coordinate_size)
            tmp_y = np.array(datafile_read[:, index_y]).reshape(self.x_coordinate_size, self.y_coordinate_size)
            self.x_coordinate_matrix = np.linspace(0, np.max(tmp_x) - np.min(tmp_x), self.u_velocity_matrix.shape[1])
            self.y_coordinate_matrix = np.linspace(0, np.max(tmp_y) - np.min(tmp_y), self.u_velocity_matrix.shape[0])

            self.normalization_flag = False
            self.normalization_direction = 'None'

            # COMMON TO ALL DATA
            self.x_coordinate_step = round((np.max(self.x_coordinate_matrix) - np.min(self.x_coordinate_matrix)) / (
                    np.size(self.x_coordinate_matrix) - 1), 32)
            self.y_coordinate_step = round((np.max(self.y_coordinate_matrix) - np.min(self.y_coordinate_matrix)) / (
                    np.size(self.y_coordinate_matrix) - 1), 32)
            self.z_coordinate_step = 0.0

            self.derivative = {'dudx': np.zeros_like(self.u_velocity_matrix),
                               'dudy': np.zeros_like(self.u_velocity_matrix),
                               'dudz': np.zeros_like(self.u_velocity_matrix),
                               'dvdx': np.zeros_like(self.u_velocity_matrix),
                               'dvdy': np.zeros_like(self.u_velocity_matrix),
                               'dvdz': np.zeros_like(self.u_velocity_matrix),
                               'dwdx': np.zeros_like(self.u_velocity_matrix),
                               'dwdy': np.zeros_like(self.u_velocity_matrix),
                               'dwdz': np.zeros_like(self.u_velocity_matrix)}

        else:
            sys.exit("\nReading error. Maybe a wrong file type?\n")

