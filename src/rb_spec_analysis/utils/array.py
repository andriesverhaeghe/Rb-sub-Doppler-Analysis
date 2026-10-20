from numpy import ndarray


def zero(array: ndarray) -> ndarray:
    """
    Assuming a 1D monotonic (increasing) array, subtract the first element
    from each element so that it starts from 0

    Args:
        array (ndarray): The array to zero

    Returns:
        zeroed array

    Raises:
        Value error if shape has len > 1
    """
    if len(array.shape) > 1:
        raise ValueError("Zero not supported for array dimensionality > 2")

    return array - array[0]
