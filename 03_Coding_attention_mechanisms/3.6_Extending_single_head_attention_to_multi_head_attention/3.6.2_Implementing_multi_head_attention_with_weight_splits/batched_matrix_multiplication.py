import torch


# (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])


print("\nTransposed Matrix:\n", a.transpose(2, 3))
print("\nTransposed Matrix Shape:\n", a.transpose(2, 3).shape) # (1, 2, 4, 3)

print("\nMultiplication Result:\n", a @ a.transpose(2, 3))


first_head = a[0, 0, :, :]
print("\nFirst Head:\n", first_head)
first_res = first_head @ first_head.T
print("\nFirst Head Result:\n", first_res)

second_head = a[0, 1, :, :]
print("\nSecond Head:\n", second_head)
second_res = second_head @ second_head.T
print("\nSecond Head Result:\n", second_res)