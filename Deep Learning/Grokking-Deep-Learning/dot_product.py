def dot_product(vec_a, vec_b):
    """
    Calculate the dot product of two vectors.

    Args:
        vec_a (list or tuple): The first vector.
        vec_b (list or tuple): The second vector.

    Returns:
        float or int: The dot product of the two vectors.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length for dot product.")
    
    # Elementwise multiplication and sum to compute the dot product
    return sum(a * b for a, b in zip(vec_a, vec_b))

# Example usage
vec_a = [1, 2, 3]
vec_b = [4, 5, 6]
result = dot_product(vec_a, vec_b)
print(result)  # Output: 32 (1*4 + 2*5 + 3*6)

