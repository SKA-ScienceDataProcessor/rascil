""" From https://goshippo.com/blog/measure-real-size-any-python-object/

"""

__all__ = ['get_size']

from distributed.protocol import pickle

def get_size(obj):
    """ Return size of object in bytes

    :param obj:
    :return:
    """
    return len(pickle.dumps(obj))
