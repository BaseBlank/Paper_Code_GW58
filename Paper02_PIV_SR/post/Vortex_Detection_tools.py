"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
The link to the reference code repository is as follows:
    https://github.com/guilindner/VortexFitting
"""


import sys
import numpy as np
import scipy.ndimage
import scipy.optimize as opt
import matplotlib.pyplot as plt


def ASCII2D_Convert_matrix(ASCII2D, matrix_H, matrix_W, index):
    """

    Args:
        ASCII2D: the original 2D ASCII file
        matrix_H: the number of rows of the matrix
        matrix_W: the number of columns of the matrix
        index: the index of the column to be extracted

    Returns: the matrix of the specified column

    """
    matrix = np.empty((matrix_H, matrix_W), dtype=np.float32)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            matrix[y, x] = ASCII2D[y * matrix.shape[1] + x, index]
    return matrix


def normalize(velocity_matrix, homogeneous_axis):
    """
    Normalize with swirling strength

    :param velocity_matrix: velocity field
    :type velocity_matrix: ndarray
    :param homogeneous_axis: False, 'x', or 'y'. The axis which the mean is subtracted
    :type homogeneous_axis: str

    :returns: normalized array
    :rtype: ndarray
    """
    if homogeneous_axis is None:
        velocity_matrix = velocity_matrix / np.sqrt(np.mean(velocity_matrix ** 2))
    elif homogeneous_axis == 'x':
        velocity_matrix = velocity_matrix / np.sqrt(np.mean(velocity_matrix ** 2, axis=1))
    elif homogeneous_axis == 'y':
        velocity_matrix = velocity_matrix / np.sqrt(np.mean(velocity_matrix ** 2, axis=0))
    else:
        sys.exit('Invalid homogeneity axis.')
    return velocity_matrix


def find_peaks(data, threshold, box_size):
    """
    Find local peaks in an image that are above a specified threshold value.

    Peaks are the maxima above the "threshold" within a local region.
    The regions are defined by the "box_size" parameters.
    "box_size" defines the local region around each pixel as a square box.

    :param data: The 2D array of the data.
    :param threshold: The data value or pixel-wise data values to be used for the detection threshold.
                      A 2D "threshold" must have the same shape as "data".
    :param box_size: The size of the local region to search for peaks at every point
    :type data: ndarray
    :type threshold: float
    :type box_size: int

    :returns: An array containing the x and y pixel location of the peaks and their values.
    :rtype: list
    """

    # .flat将数组转换为1-D的迭代器，迭代器flat返回每一个元素, np.all()判断所有元素是否为True
    if np.all(data == data.flat[0]):
        return []

    data_max = scipy.ndimage.maximum_filter(data, size=box_size, mode='constant', cval=0.0)

    peak_goodmask = (data == data_max)  # good pixels are True, boolean array

    # np.logical_and(), Returns X1 and X2 with the logical Boolean value
    peak_goodmask = np.logical_and(peak_goodmask, (data > threshold))  # boolean array
    y_peaks, x_peaks = peak_goodmask.nonzero()
    peak_values = data[y_peaks, x_peaks]
    peaks = (y_peaks, x_peaks, peak_values)  # y index, x index, values
    return peaks


def direction_rotation(vorticity, peaks):
    """
    Identify the direction of the vortices rotation using the vorticity.

    :param vorticity: 2D array with the computed vorticity
    :param peaks: list of the detected peaks
    :type vorticity: ndarray
    :type peaks: list

    :returns: vortices_clockwise, vortices_counterclockwise, arrays containing the direction of rotation for each vortex
    :rtype: list
    """

    vortices_clockwise_x, vortices_clockwise_y, vortices_clockwise_cpt = [], [], []  # clockwise rotation
    vortices_counterclockwise_x, vortices_counterclockwise_y, vortices_counterclockwise_cpt = [], [], []  # counterclockwise rotation
    for i in range(len(peaks[0])):
        if vorticity[peaks[0][i], peaks[1][i]] > 0.0:  # Spin flow
            vortices_clockwise_y.append(peaks[0][i])
            vortices_clockwise_x.append(peaks[1][i])
            vortices_clockwise_cpt.append(peaks[2][i])
        else:  # Spinless flow
            vortices_counterclockwise_y.append(peaks[0][i])
            vortices_counterclockwise_x.append(peaks[1][i])
            vortices_counterclockwise_cpt.append(peaks[2][i])
    vortices_clockwise = (vortices_clockwise_y, vortices_clockwise_x, vortices_clockwise_cpt)
    vortices_counterclockwise = (vortices_counterclockwise_y, vortices_counterclockwise_x, vortices_counterclockwise_cpt)
    vortices_clockwise = np.asarray(vortices_clockwise)  # [[y], [x], [values]]
    vortices_counterclockwise = np.asarray(vortices_counterclockwise)  # [[y], [x], [values]]
    return vortices_clockwise, vortices_counterclockwise


def get_vortices(x_coordinate_matrix, y_coordinate_matrix,
                 x_coordinate_step, y_coordinate_step,
                 u_velocity_matrix, v_velocity_matrix,
                 peaks, vorticity, rmax, correlation_threshold):
    """
    General routine to check if the detected vortex is a real vortex

    :param vfield: data from the input file
    :param peaks: list of vortices
    :param vorticity: calculated field
    :param rmax: maximum radius (adapt it to your data domain)
    :param correlation_threshold: threshold to detect a vortex (default is 0.75)
    :type vfield: class VelocityField
    :type peaks: list
    :type vorticity: ndarray
    :type rmax: float
    :type correlation_threshold: float
    :returns: list of detected vortices
    :rtype: list
    """

    vortices = list()
    cpt_accepted = 0
    dx = x_coordinate_step
    dy = y_coordinate_step
    for i in range(len(peaks[0])):  # 对每一个peak点进行处理
        y_center_index = peaks[0][i]
        x_center_index = peaks[1][i]
        print(i, 'Processing detected swirling at (x, y)', x_center_index, y_center_index)
        if rmax == 0.0:
            core_radius = 2 * np.hypot(dx, dy)  # 半径, hypot() 返回欧几里德范数 sqrt(x*x + y*y)
        else:
            core_radius = rmax  # guess on the starting vortex radius
        # circulation contained in the vortex
        gamma = vorticity[y_center_index, x_center_index] * np.pi * core_radius ** 2  # 旋度乘以圆的面积

        # [core_radius, gamma, x_real, y_real, u_advection, v_advection, dist]
        # The parameters of the vortex are obtained by fitting the vortex.
        vortices_parameters = full_fit(core_radius, gamma,
                                       x_coordinate_matrix, y_coordinate_matrix,
                                       x_coordinate_step, y_coordinate_step,
                                       u_velocity_matrix, v_velocity_matrix,
                                       x_center_index, y_center_index)
        if vortices_parameters[6] < 2:  # dist
            correlation_value = 0
        else:
            x_index, y_index, u_data, v_data = window(u_velocity_matrix, v_velocity_matrix,
                                                      x_coordinate_matrix, y_coordinate_matrix,
                                                      round((vortices_parameters[2] - np.min(x_coordinate_matrix)) / dx, 0),
                                                      round((vortices_parameters[3] - np.min(y_coordinate_matrix)) / dy, 0),
                                                      vortices_parameters[6])  # [-dist, dist], 2D array
            u_model, v_model = velocity_model(vortices_parameters[0], vortices_parameters[1],
                                              vortices_parameters[2], vortices_parameters[3],
                                              vortices_parameters[4], vortices_parameters[5],
                                              x_index, y_index)  # The fitted u and v
            correlation_value = correlation_coef(u_data - vortices_parameters[4], v_data - vortices_parameters[5],
                                                 u_model - vortices_parameters[4], v_model - vortices_parameters[5])
        if correlation_value > correlation_threshold:
            print('Accepted! Correlation = {:1.2f} (vortex #{:2d})'.format(correlation_value, cpt_accepted))
            # compute the tangential velocity at critical radius
            u_theta = (vortices_parameters[1] / (2 * np.pi * vortices_parameters[0])) * (1 - np.exp(-1))
            vortices.append(
                [vortices_parameters[0], vortices_parameters[1], vortices_parameters[2], vortices_parameters[3],
                 vortices_parameters[4],
                 vortices_parameters[5], vortices_parameters[6], correlation_value, u_theta])
            cpt_accepted += 1
    return vortices


def full_fit(core_radius, gamma,
             x_coordinate_matrix, y_coordinate_matrix,
             x_coordinate_step, y_coordinate_step,
             u_velocity_matrix, v_velocity_matrix,
             x_center_index, y_center_index):
    """Full fitting procedure, calculate return vortices parameters

    :param core_radius: core radius of the vortex
    :param gamma: circulation contained in the vortex
    :param vfield: data from the input file
    :param x_center_index: x matrix index of the vortex center
    :param y_center_index: y matrix index of the vortex center
    :type core_radius: float
    :type gamma: float
    :type vfield: class
    :type x_center_index: int
    :type y_center_index: int
    :returns: fitted[i], dist
    :rtype: list
    """

    fitted = [[], [], [], [], [], []]  # core_radius, gamma, x_center, y_center, u_center, v_center
    fitted[0] = core_radius
    fitted[1] = gamma
    fitted[2] = x_coordinate_matrix[x_center_index]  # x-coordinate
    fitted[3] = y_coordinate_matrix[y_center_index]  # y-coordinate
    dx = x_coordinate_step
    dy = y_coordinate_step
    dist = 0
    # correlation_value = 0.0
    for i in range(50):  # i是fit参数的迭代次数，default=10
        # The coordinates of own data do not start from 0
        x_center_index = int(round((fitted[2] - np.min(x_coordinate_matrix)) / dx))  # x_coordinate_matrix, x coordinates do not start at 0
        y_center_index = int(round((fitted[3] - np.min(y_coordinate_matrix)) / dy))  # y_coordinate_matrix, y coordinates do not start at 0
        if x_center_index >= u_velocity_matrix.shape[1]:
            x_center_index = u_velocity_matrix.shape[1] - 1
        if x_center_index <= 2:
            x_center_index = 3
        if y_center_index >= v_velocity_matrix.shape[0]:
            y_center_index = v_velocity_matrix.shape[0] - 1
        r1 = fitted[0]
        x1 = fitted[2]
        y1 = fitted[3]
        dist = int(round(fitted[0] / np.hypot(dx, dy), 0)) + 1
        if fitted[0] < 2 * np.hypot(dx, dy):  # 涡旋半径小于2倍dx, dy对角长
            break
        fitted[4] = u_velocity_matrix[y_center_index, x_center_index]  # u_advection
        fitted[5] = v_velocity_matrix[y_center_index, x_center_index]  # v_advection
        x_index, y_index, u_data, v_data = window(u_velocity_matrix, v_velocity_matrix,
                                                  x_coordinate_matrix, y_coordinate_matrix,
                                                  x_center_index, y_center_index, dist)  # -dist, dist

        # shape=(6,), [core_radius, gamma, x_real, y_real, u_advection, v_advection]
        fitted = fit(fitted[0], fitted[1], x_index, y_index, fitted[2], fitted[3],
                     u_data, v_data, fitted[4], fitted[5], i)
        if i > 0:
            # break if radius variation is less than 10% and accepts 如果半径变化小于10%，则断裂，并接受
            if abs(fitted[0] / r1 - 1) < 0.1:
                if (abs((fitted[2] / x1 - 1)) < 0.1) or (abs((fitted[3] / y1 - 1)) < 0.1):
                    break
            # break if x or y position is out of the window and discards 如果X或Y的位置超出了窗口，则断开，并丢弃
            if (abs((fitted[2] - x1)) > dist * dx) or (abs((fitted[3] - y1)) > dist * dy):
                dist = 0
                break
    return fitted[0], fitted[1], fitted[2], fitted[3], fitted[4], fitted[5], dist


def window(u_velocity_matrix, v_velocity_matrix,
           x_coordinate_matrix, y_coordinate_matrix,
           x_center_index, y_center_index, dist):
    """
    Defines a window around (x; y) coordinates, window Both length and style are dist, -dist ~ dist

    :param vfield: full size velocity field
    :type vfield: ndarray
    :param x_center_index: box center index (x)
    :type x_center_index: int
    :param y_center_index: box center index (y)
    :type y_center_index: int
    :param dist: size of the vortex (mesh units)
    :param dist: int

    :returns: cropped arrays for x, y, u and v
    :rtype: 2D arrays of floats

    """
    if x_center_index - dist > 0:
        x1 = x_center_index - dist
    else:
        x1 = 0
    if y_center_index - dist > 0:
        y1 = y_center_index - dist
    else:
        y1 = 0
    if x_center_index + dist <= u_velocity_matrix.shape[1]:
        x2 = x_center_index + dist
    else:
        x2 = u_velocity_matrix.shape[1]
    if y_center_index + dist <= v_velocity_matrix.shape[0]:
        y2 = y_center_index + dist
    else:
        y2 = v_velocity_matrix.shape[0]
    x_index, y_index = np.meshgrid(x_coordinate_matrix[int(x1):int(x2)],
                                   y_coordinate_matrix[int(y1):int(y2)],
                                   indexing='xy')
    u_data = u_velocity_matrix[int(y1):int(y2), int(x1):int(x2)]
    v_data = v_velocity_matrix[int(y1):int(y2), int(x1):int(x2)]
    return x_index, y_index, u_data, v_data


# fitted[0], fitted[1], x_index, y_index, fitted[2], fitted[3], u_data, v_data, fitted[4], fitted[5], i
def fit(core_radius, gamma, x, y, x_real, y_real, u_data, v_data, u_advection, v_advection, i):
    """
    Fitting of the Lamb-Oseen Vortex

    :param core_radius: core radius of the vortex
    :param gamma: circulation contained in the vortex
    :param x: x position window
    :param y: y position window
    :param x_real: x position of the vortex center
    :param y_real: y position of the vortex center
    :param u_data: velocity u from the data at the proposed window
    :param v_data: velocity v from the data at the proposed window
    :param u_advection: uniform advection velocity u
    :param v_advection: uniform advection velocity u
    :param i: current iteration for fitting
    :type core_radius: float
    :type gamma: float
    :type x: ndarray
    :type y: ndarray
    :type x_real: float
    :type y_real: float
    :type u_data: ndarray
    :type v_data: ndarray
    :type u_advection: float
    :type v_advection: float
    :type i: iterator
    :returns: fitted parameters (core_radius, gamma,xcenter,ycenter, u_advection, v_advection...)
    :rtype: list
    """
    # Method for opt.least_squares fitting. Can be 'trf', 'dogbox' or 'lm'.
    # 'trf': Trust Region Reflective algorithm
    # 'dogbox': dogleg algorithm
    # 'lm': Levenberg-Marquardt algorithm
    method = 'trf'

    x = x.ravel()  # 是将原数组拉伸成为一维数组, 与flatten的区别是flatten返回的是拷贝，不修改原值
    y = y.ravel()
    u_data = u_data.ravel()
    v_data = v_data.ravel()
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    def lamb_oseen_model(fitted):
        """
        Lamb-Oseen velocity model used for the nonlinear fitting 用于非线性拟合的 Lamb-Oseen 速度模型

        :param fitted: parameters of a vortex (core_radius, gamma,xcenter,ycenter, u_advection, v_advection)
        :type fitted: list
        :returns: velocity field, following a Lamb-Oseen model
        :rtype: ndarray
        """

        core_radius_model = fitted[0]  # core_radius
        gamma_model = fitted[1]  # gamma
        xcenter_model = fitted[2]  # x_real
        ycenter_model = fitted[3]  # y_real
        u_advection_model = fitted[4]  # u_advection
        v_advection_model = fitted[5]  # v_advection
        r = np.hypot(x - xcenter_model, y - ycenter_model)  # x.ravel(), so one-dimensional numpy array, shape=(2*dist)**2
        # Lamb–Oseen vortex
        u_theta_model = gamma_model / (2 * np.pi * r) * (1 - np.exp(-r ** 2 / core_radius_model ** 2))
        u_theta_model = np.nan_to_num(u_theta_model)  # 用零替换NaN，用最大的有限数替换无穷大
        u_model = u_advection_model - u_theta_model * (y - ycenter_model) / r - u_data
        v_model = v_advection_model + u_theta_model * (x - xcenter_model) / r - v_data
        u_model = np.nan_to_num(u_model)
        v_model = np.nan_to_num(v_model)
        vfield_model = np.append(u_model, v_model)  # one-dimensional numpy array, When the axis is not specified, shape=2*(2*dist)**2
        return vfield_model

    if i > 0:
        m = 1.0
    else:
        m = 4.0

    if method == 'trf':
        epsilon = 0.001  # 希腊字母
        bnds = ([0, gamma - abs(gamma) * m / 2 - epsilon, x_real - m * dx - epsilon, y_real - m * dy - epsilon,
                 u_advection - abs(u_advection) - epsilon, v_advection - abs(v_advection) - epsilon],
                [core_radius + core_radius * m, gamma + abs(gamma) * m / 2 + epsilon, x_real + m * dx + epsilon,
                 y_real + m * dy + epsilon, u_advection + abs(u_advection) + epsilon,
                 v_advection + abs(v_advection) + epsilon])

        # lamb_oseen_model本身就是残差函数，
        # [core_radius, gamma, x_real, y_real, u_advection, v_advection]是要拟合的曲线参数，不是y值
        # lamb oseen模型的参数是core_radius, gamma, x_real, y_real, u_advection, v_advection
        # opt.least_squares反复计算几次，知道最小的拟合误差
        sol = opt.least_squares(lamb_oseen_model, [core_radius, gamma, x_real, y_real, u_advection, v_advection],
                                method='trf', bounds=bnds)  # 有界的拟合
    elif method == 'dogbox':
        epsilon = 0.001
        bnds = ([0, gamma - abs(gamma) * m / 2 - epsilon, x_real - m * dx - epsilon, y_real - m * dy - epsilon,
                 u_advection - abs(u_advection) - epsilon, v_advection - abs(v_advection) - epsilon],
                [core_radius + core_radius * m, gamma + abs(gamma) * m / 2 + epsilon, x_real + m * dx + epsilon,
                 y_real + m * dy + epsilon, u_advection + abs(u_advection) + epsilon,
                 v_advection + abs(v_advection) + epsilon])

        sol = opt.least_squares(lamb_oseen_model, [core_radius, gamma, x_real, y_real, u_advection, v_advection],
                                method='dogbox', bounds=bnds)
    elif method == 'lm':
        sol = opt.least_squares(lamb_oseen_model, [core_radius, gamma, x_real, y_real, u_advection, v_advection],
                                method='lm', xtol=10 * np.hypot(dx, dy))

    return sol.x  # shape=(6,), [core_radius, gamma, x_real, y_real, u_advection, v_advection]


def velocity_model(core_radius, gamma, x_real, y_real, u_advection, v_advection, x, y):
    """Generates the Lamb-Oseen vortex velocity array

    :param core_radius: core radius of the vortex
    :param gamma: circulation contained in the vortex
    :param x_real: relative x position of the vortex center
    :param y_real: relative y position of the vortex center
    :param u_advection: u advective velocity at the center
    :param v_advection: v advective velocity at the center
    :param x: x_index, 2D array
    :param y: y_index, 2D array
    :type core_radius: float
    :type gamma: float
    :type x_real: float
    :type y_real: float
    :type u_advection: float
    :type v_advection: float
    :type x: float
    :type y: float
    :returns: velx, vely
    :rtype: float
    """
    r = np.hypot(x - x_real, y - y_real)  # 2D array, (236, 116)
    vel = (gamma / (2 * np.pi * r)) * (1 - np.exp(-(r ** 2) / core_radius ** 2))
    vel = np.nan_to_num(vel)
    velx = u_advection - vel * (y - y_real) / r
    vely = v_advection + vel * (x - x_real) / r
    velx = np.nan_to_num(velx)
    vely = np.nan_to_num(vely)
    # print(core_radius, gamma, x_real, y_real, u_advection, v_advection, x, y)
    return velx, vely


def correlation_coef(u_data, v_data, u_model, v_model):
    """Calculates the correlation coefficient between two 2D arrays

    :param u_data: velocity u from the data at the proposed window
    :param v_data: velocity v from the data at the proposed window
    :param u_model: velocity u from the calculated model
    :param v_model: velocity v from the calculated model
    :type u_data: ndarray
    :type v_data: ndarray
    :type u_model: ndarray
    :type v_model: ndarray
    :returns: correlation
    :rtype: float
    """
    u_data = u_data.ravel()
    v_data = v_data.ravel()
    u = u_model.ravel()
    v = v_model.ravel()

    prod_piv_mod = np.mean(u_data * u + v_data * v)
    prod_piv = np.mean(u * u + v * v)
    prod_mod = np.mean(u_data * u_data + v_data * v_data)
    correlation = prod_piv_mod / (max(prod_piv, prod_mod))

    return correlation


