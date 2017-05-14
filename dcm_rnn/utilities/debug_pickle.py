import pickle

def debug_pickle(instance):
    """
    :return: Which attribute from this object can't be pickled?
    """
    attribute = None

    for k, v in instance.__dict__.iteritems():
        try:
            pickle.dumps(v)
        except:
            attribute = k
            break

    return attribute