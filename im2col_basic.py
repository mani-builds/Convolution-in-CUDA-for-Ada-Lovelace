import numpy as np
import math

def dim_output(size_of_input, P, K, S):
    return math.floor((size_of_input + 2*P - K) / S ) + 1

image = np.array([i for i in range(0,3*3*3)])
image = image.reshape([3,3,3]) # channel, row, col

filter = np.array([i for i in range(0,2*2*3)])
filter = filter.reshape([3,2,2])
print(f"image's shape: {image.shape}")
print(image)
print(f"filter's shape: {filter.shape}")
print(filter)


stride = 1 # how many pixels the filter jumps each time
padding = 0 # the number of zero-pixel borders added
H_in = image.shape[1]
W_in = image.shape[2]

print(f"IN DIMS: {H_in}, {W_in}")
kernel_size = filter.shape[1]
patch_size = filter.shape[1] * filter.shape[2] * filter.shape[0]

H_out = dim_output(H_in, padding, kernel_size, stride) # No. of position for Height
W_out = dim_output(W_in, padding, kernel_size, stride) # No. of position for Width

print(f"OUT DIMS: {H_out}, {W_out}")
num_of_patches = H_out * W_out

im2col_image = np.zeros(patch_size*num_of_patches)
im2col_image = im2col_image.reshape([patch_size, num_of_patches])

# Flatten input image
for each_patch in range(num_of_patches):
    # 1. Calculate the top-left corner of the current patch
    # This converts 1D patch index into 2D (row, col) coordinates
    start_h = (each_patch // W_out) * stride
    start_w = (each_patch % W_out) * stride

    idx_count = 0
    for channel in range(filter.shape[0]):
        for row in range(filter.shape[1]):
            for col in range(filter.shape[2]):
                # 2. Use the start coordinates to "offset" the selection
                im2col_image[idx_count][each_patch] = image[channel, start_h + row, start_w + col]
                idx_count += 1

print(im2col_image)
print("Shape of im2col_image: ", im2col_image.shape)

# Flatten kernel
filter_flattened = filter.reshape([1,12])
print(filter_flattened)
print("Shape of filter_flattened: ", filter_flattened.shape)

# output = flattened_filter * im2col_image
output = np.dot(filter_flattened, im2col_image)
output = output.reshape([H_out,W_out])
print(output)
print("Shape of output: ", output.shape)
