def vector_average(vec_a):
    """
    Calculate the average of all elements in a vector.

    Args:
        vec_a (list or tuple): The input vector.

    Returns:
        float: The average of all elements in the vector.

    Raises:
        ValueError: If the vector is empty.
    """
    if len(vec_a) == 0:
        raise ValueError("Vector is empty, cannot compute average.")
    
    return sum(vec_a) / len(vec_a)

# Example usage
vec_a = [1, 2, 3, 4]
result = vector_average(vec_a)
print(result)  # Output: 2.5

