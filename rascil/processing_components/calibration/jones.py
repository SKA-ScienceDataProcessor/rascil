import numpy


def apply_jones(ej, cfs, inverse=False):
    """ Apply Jones matrix (or inverse)

    :param ej: 2x2 Jones matrix
    :param cfs: 4 vector of stokes
    :param inverse: Calculate the inverse
    :return:
    """
    cfs = cfs.reshape([2, 2])
    assert ej.shape == (2, 2)
    if inverse:
        cej = numpy.conjugate(ej).T
        inv_ej = numpy.linalg.inv(ej)
        inv_cej = numpy.linalg.inv(cej)
        return (inv_ej @ cfs @ inv_cej).reshape([4])
    else:
        cej = numpy.conjugate(ej).T
        return (ej @ cfs @ cej).reshape([4])
