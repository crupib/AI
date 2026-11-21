import torch

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

# Step 1: Calculate attention scores
attn_scores = inputs @ inputs.T

# Step 2: Normalize scores across each row
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# Step 3: Select the second input as the query
query = inputs[1]

# Step 4: Calculate the second context vector
context_vec_2 = torch.zeros_like(query)

for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights[1, i] * x_i

print("Second context vector:", context_vec_2)

# Step 5: Check the example row sum
row_2_sum = sum(
    [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]
)
print("Row 2 sum:", row_2_sum)

# Step 6: Check all row sums
print("All row sums:", attn_weights.sum(dim=-1))

# Step 7: Calculate all context vectors
all_context_vecs = attn_weights @ inputs

print("All context vectors:")
print(all_context_vecs)

print("Previous 2nd context vector:", context_vec_2)

# Step 8: Verify both calculations agree
print(
    "Results match:",
    torch.allclose(all_context_vecs[1], context_vec_2)
)

