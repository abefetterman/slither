import math

# functions to compute output layer size for conv nets:
# equation for number of dimensions
def conv_eq(dim_in, padding, dilation, kernel_size, stride):
    return math.floor((dim_in + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1)

# computes output size for single layer
def conv_size(dims,conv):
    dims_out=[]
    for i in range(len(dims)):
        dim = conv_eq(dims[i], conv.padding[i], conv.dilation[i], conv.kernel_size[i], conv.stride[i])
        dims_out.append(dim)
    return tuple(dims_out)

# computes output size for chain conv nets
def conv_chain(dims_in,conv_list):
    dims = dims_in
    for conv in conv_list:
        dims=conv_size(dims,conv)
    return dims
