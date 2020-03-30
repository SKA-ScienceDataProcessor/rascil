import numpy


def apply_jones(ej, cfs, inverse=False, min_det=1e-6):
    """ Apply Jones matrix (or inverse)

    :param ej: 2x2 Jones matrix
    :param cfs: 2x2 matrix of stokes
    :param inverse: Calculate the inverse
    :param min_det: Minimum determinant in invert
    :return:
    """
    if inverse:
        if numpy.abs(numpy.linalg.det(ej)) > min_det:
            inv_ej = numpy.linalg.inv(ej)
            inv_cej = numpy.conjugate(inv_ej).T
            return inv_ej @ cfs @ inv_cej
        else:
            return 0.0 * cfs
    else:
        cej = numpy.conjugate(ej).T
        return ej @ cfs @ cej
