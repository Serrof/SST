import numpy as np

# default parameters
default_nb_terms = 100
default_threshold = 1.e10


def preprocess(x_m, y_m, s_x, s_y):
    """Vectorized function to sort arguments of collision probability, as s_x is assumed smaller or equal to s_y in the
    LAAS formula.

            Args:
                x_m (numpy array): abscissa of mean relative position
                y_m (numpy array): ordinate of mean relative position
                s_x (numpy array): standard-deviation of abscissa
                s_y (numpy array): standard-deviation of ordinate

            Returns:
                (numpy array): new abscissa
                (numpy array): new ordinate
                (numpy array): new standard-deviation of abscissa
                (numpy array): new standard-deviation of ordinate
    """

    # copy inputs
    new_x = np.array(x_m)
    new_y = np.array(y_m)
    s_small = np.array(s_x)
    s_big = np.array(s_y)

    # fetch indices to swap
    indices = s_x - s_y > 0.

    # process abscissa and ordinate where necessary
    new_x[indices] = -y_m[indices]
    new_y[indices] = x_m[indices]
    s_small[indices] = s_y[indices]
    s_big[indices] = s_x[indices]

    return new_x, new_y, s_small, s_big


def evaluate(r, x_m, y_m, s_x, s_y, n=default_nb_terms, threshold=default_threshold):
    """Vectorized function to compute so-called short-term collision probabilities i.e. when the relative motion is
    approximated as uniform rectilinear with negligible uncertainty on velocity (and the two objects are modelled as
    spheres). The corresponding 2-D Gaussian integral is calculated here with the formula developed at LAAS-CNRS: Fast
    and Accurate Computation of Orbital Collision Probability for Short-Term Encounters, Jan. 2016, JGCD 39(5):1-13
    DOI: 10.2514/1.G001353. It boils down to the product between an exponential and a power series with positive terms
    computed recursively.

            Args:
                r (numpy array): radius of combined object
                x_m (numpy array): abscissa of mean relative position
                y_m (numpy array): ordinate of mean relative position
                s_x (numpy array): standard-deviation of abscissa
                s_y (numpy array): standard-deviation of ordinate
                n (int): number of terms to be used when calculation the power series involved in the computation of the
                collision probability. One hundred is usually well enough for typical encounters. You would need more
                when r >> s_x. There exists a conservative estimate for n but it is not implemented here.
                threshold (float): value used to scale summed terms in series to avoid numerical overflow

            Returns:
                (numpy array): collision probability
    """

    # the formula assumes that s_x <= s_y, so if not the case, arguments are re-arranged
    if len(s_y[s_x - s_y > 0.]) > 0:
        new_x, new_y, s_small, s_big = preprocess(x_m, y_m, s_x, s_y)
        return evaluate(r, new_x, new_y, s_small, s_big, n, threshold)

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
    omega_y = (ratio_y / s_y)**2 / 4.
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

    exponent = np.zeros(len(r))

    # sum over terms
    sc = np.zeros(len(r))
    terms = [c_0, c_1, c_2, c_3]
    for k in range(0, min(n, 4)):
        sc += terms[k]

        # if necessary, rescale quantities
        indices = sc > threshold
        exponent[indices] += np.log10(threshold)
        c_0[indices] /= threshold
        c_1[indices] /= threshold
        c_2[indices] /= threshold
        c_3[indices] /= threshold
        sc[indices] /= threshold

    aux0 = r6 * p3 * phi_y2 * omega_x
    aux1 = r4 * p2 * phi_y
    aux2 = 2. * omega_x * inter0
    aux3 = phi_y * (2. * omega_x + 3. * p / 2.) + omega
    aux4 = p_times_phi * inter0 * 2.
    aux5 = p * (2. * phi_y + 1)

    k_plus_2, k_plus_3, k_plus_4, k_plus_5 = 2., 3., 4., 5.
    halfy = 2.5

    # iterate

    for __ in range(0, n - 4):

        # if necessary, rescale quantities
        indices = sc > threshold
        exponent[indices] += np.log10(threshold)
        c_0[indices] /= threshold
        c_1[indices] /= threshold
        c_2[indices] /= threshold
        c_3[indices] /= threshold
        sc[indices] /= threshold

        # recurrence relation
        inter = c_3 * (inter1 + aux5 * k_plus_3)
        inter -= c_2 * p_times_r2 * (aux4 * halfy + aux3) / k_plus_4
        denom = k_plus_4 * k_plus_3
        inter += c_1 * aux1 * (p_times_phi * halfy + aux2) / denom
        inter -= c_0 * aux0 / (denom * k_plus_2)
        inter *= r2 / (k_plus_4 * k_plus_5)
        c_0, c_1, c_2, c_3 = c_1, c_2, c_3, inter

        # update intermediate variables
        k_plus_2, k_plus_3, k_plus_4, k_plus_5 = k_plus_3, k_plus_4, k_plus_5, k_plus_5 + 1.
        halfy += 1.

        # update sum
        sc += c_3

    # compute actual collision probability
    return sc * np.exp(exponent * np.log(10.) - p_times_r2)


def bound(r, x_m, y_m, s_x, s_y):
    """Vectorized function to compute lower and upper bounds on the short-term collision probability. These analytical
    bounds are a by-product from the LAAS formula. For real alerts, they usually already give the order of magnitude for
    the probability and maybe even some of the first digits.

            Args:
                r (numpy array): radius of combined object
                x_m (numpy array): abscissa of mean relative position
                y_m (numpy array): ordinate of mean relative position
                s_x (numpy array): standard-deviation of abscissa
                s_y (numpy array): standard-deviation of ordinate

            Returns:
                (numpy array): lower bound on collision probability
                (numpy array): upper bound on collision probability
    """

    # the formula assumes that s_x <= s_y, so if not the case, arguments are re-arranged
    if len(s_y[s_x - s_y > 0.]) > 0:
        new_x, new_y, s_small, s_big = preprocess(x_m, y_m, s_x, s_y)
        return bound(r, new_x, new_y, s_small, s_big)

    # pre-computations
    r2 = r ** 2
    p = 1. / (2. * s_x ** 2)
    p_times_r2 = p * r2
    phi_y = 1. - (s_x / s_y) ** 2
    ratio_x = x_m / s_x
    ratio_y = y_m / s_y
    omega_x = (ratio_x / s_x) ** 2 / 4.
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

    print(evaluate(Rs, Xs, Ys, SXs, SYs, 100))
    print(bound(Rs, Xs, Ys, SXs, SYs))
