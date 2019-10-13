import numpy as np

# default parameters
default_nb_terms = 100
default_threshold = 1.e10


def evaluate(r, x_m, y_m, s_x, s_y, n=default_nb_terms, threshold=default_threshold):
    """Function to compute so-called short-term collision probabilities i.e. when the relative motion is approximated as
    uniform rectilinear with negligible uncertainty on velocity (and the two objects are modelled as spheres).
    The corresponding 2-D Gaussian integral is calculated here with the formula developed at LAAS-CNRS in 2014-2015,
    boiling down to the product between an exponential and a power series with positive terms computed recursively.

            Args:
                r (float or numpy array): radius of combined object
                x_m (float or numpy array): abscissa of mean relative position
                y_m (float or numpy array): ordinate of mean relative position
                s_x (float or numpy array): standard-deviation of abscissa
                s_y (float or numpy array): standard-deviation of ordinate
                n (int): number of terms to be used when calculation the power series involved in the computation of the
                collision probability. One hundred is usually well enough for typical encounters. You would need more
                when r >> s_x. There exists a conservative estimate for n but it is not implemented here.
                threshold (float): value used to scale summed terms in series to avoid numerical overflow

            Returns:
                (float or numpy array): collision probability
    """

    # pre-computations

    r2 = r ** 2
    r4 = r2**2
    r6 = r4 * r2

    p = 1. / (2. * s_x**2)
    p2 = p * p
    p3 = p2 * p
    p_times_r2 = p * r2

    phi_y = 1. - (s_x / s_y)**2
    phi_y2 = phi_y ** 2
    phi_y3 = phi_y * phi_y2
    p_times_phi = p * phi_y

    ratio_x = x_m / s_x
    ratio_y = y_m / s_y
    omega_x = (ratio_x / s_x)**2 / 4.
    omega_y = (ratio_y / s_y) ** 2 / 4.
    omega = omega_x + omega_y

    alpha_0 = np.exp(-(ratio_x**2 + ratio_y**2) / 2.) / (2. * s_x * s_y)

    # initialize recurrence

    inter0 = 1. + phi_y / 2.
    inter1 = p * inter0 + omega
    inter2 = p * (p * (1. + phi_y**2 / 2.) + 2. * phi_y * omega_y)
    inter3 = inter1 ** 2
    c_0 = alpha_0 * r2
    c_1 = c_0 * (r2 / 2.) * inter1
    c_2 = c_0 * (r4 / 12.) * (inter3 + inter2)
    c_3 = c_0 * (r6 / 144.) * (inter1 * (inter3 + 3. * inter2) + 2. * (p3 * (1. + phi_y3 / 2.) +
                                                                         3. * p2 * phi_y2 * omega_y))

    s = c_0 + c_1 + c_2 + c_3

    aux0 = r6 * p3 * phi_y2 * omega_x
    aux1 = r4 * p2 * phi_y
    aux2 = 2. * omega_x * inter0
    aux3 = phi_y * (2. * omega_x + 3. * p / 2.) + omega
    aux4 = p_times_phi * inter0 * 2.
    aux5 = p * (2. * phi_y + 1)

    k_plus_2 = 2.
    k_plus_3 = 3.
    k_plus_4 = 4.
    k_plus_5 = 5.
    halfy = 2.5

    exponent = r * 0.

    # iterate

    for k in range(0, n - 4):

        # if necessary, rescale quantities
        indices = s > threshold
        exponent[indices] += np.log10(threshold)
        s[indices] /= threshold
        c_3[indices] /= threshold
        c_2[indices] /= threshold
        c_1[indices] /= threshold
        c_0[indices] /= threshold

        # recurrence relation
        aux = c_3 * (inter1 + aux5 * k_plus_3)
        aux -= c_2 * p_times_r2 * (aux4 * halfy + aux3) / k_plus_4
        denom = k_plus_4 * k_plus_3
        aux += c_1 * aux1 * (p_times_phi * halfy + aux2) / denom
        aux -= c_0 * aux0 / (denom * k_plus_2)
        aux *= r2 / (k_plus_4 * k_plus_5)
        c_0, c_1, c_2, c_3 = c_1, c_2, c_3, aux

        # update intermediate variables
        k_plus_2, k_plus_3, k_plus_4, k_plus_5 = k_plus_3, k_plus_4, k_plus_5, k_plus_5 + 1.
        halfy += 1.

        # update quantity of interest
        s += c_3

    return s * np.exp(exponent * np.log(10.) - p_times_r2)


def bound(r, x_m, y_m, s_x, s_y):
    """Function to compute lower and upper bounds on the short-term collision probability. These analytical bounds are
    a by-product from the LAAS formula. For real alerts, they usually already give the order of magnitude for
    the probability and maybe even some of the first digits.

            Args:
                r (float or numpy array): radius of combined object
                x_m (float or numpy array): abscissa of mean relative position
                y_m (float or numpy array): ordinate of mean relative position
                s_x (float or numpy array): standard-deviation of abscissa
                s_y (float or numpy array): standard-deviation of ordinate

            Returns:
                (float or numpy array): lower bound on collision probability
                (float or numpy array): upper bound on collision probability
    """

    # pre-computations
    r2 = r ** 2
    p = 1. / (2. * s_x**2)
    p_times_r2 = p * r2
    phi_y = 1. - (s_x / s_y)**2
    ratio_x = x_m / s_x
    ratio_y = y_m / s_y
    omega_x = (ratio_x / s_x)**2 / 4.
    omega_y = (ratio_y / s_y) ** 2 / 4.
    inter = phi_y / 2. + (omega_x + omega_y) / p
    alpha_0 = np.exp(-(ratio_x**2 + ratio_y**2) / 2.) / (2. * s_x * s_y)
    alpha_0_over_p = alpha_0 / p
    ex = -np.exp(-p_times_r2)

    # compute lower bound (always positive)
    lb = alpha_0_over_p * (1. + ex)

    # compute upper bound
    ub = alpha_0_over_p * (np.exp(inter * p_times_r2) + ex) / (1. + inter)
    ub[ub > 1.] = 1.

    return lb, ub


if __name__ == '__main__':

    Rs = np.array([5., 5., 5., 5., 10., 10., 10., 10., 10., 10., 50., 50.,
                   10.3, 1.3, 5.3, 3.5, 13.2, 15.])
    Xs = np.array([0., 10., 0., 10., 0., 1000., 0., 10000., 0., 10000., 0., 5000.,
                   84.875546, -81.618369, 102.177247, -752.672701, -692.362271, -3.8872073])
    Ys = np.array([10., 0., 10., 0., 1000., 0., 10000., 0., 10000., 0., 5000., 0.,
                   60.583685, 115.055899, 693.405893, 544.939441, 4475.456261, 0.1591646])
    SXs = np.array([25., 25., 25., 25., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000.,
                    57.918666, 15.988242, 94.230921, 445.859945, 193.454603, 1.4101830])
    SYs = np.array([50., 50., 75., 75., 3000., 3000., 3000., 3000., 10000, 10000, 3000., 3000.,
                    152.8814468, 5756.840725, 643.409272, 6095.858688, 562.027293, 114.258519])

    print(evaluate(Rs, Xs, Ys, SXs, SYs))
    print(bound(Rs, Xs, Ys, SXs, SYs))
