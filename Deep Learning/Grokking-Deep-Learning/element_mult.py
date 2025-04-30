def elementwise_multiplication(vec_a, vec_b):
    """
    Perform elementwise multiplication of two vectors.

    Args:
        vec_a (list or tuple): The first vector.
        vec_b (list or tuple): The second vector.

    Returns:
        list: A new list containing the elementwise multiplication of vec_a and vec_b.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length for elementwise multiplication.")
    
    return [a * b for a, b in zip(vec_a, vec_b)]

# Example usage
vec_a = [1, 2, 3]
vec_b = [4, 5, 6]
result = elementwise_multiplication(vec_a, vec_b)
print(result)  # Output: [4, 10, 18]

