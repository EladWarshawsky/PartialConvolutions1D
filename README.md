# PartialConv1d

`PartialConv1d` is a PyTorch module that extends the `nn.Conv1d` class and performs partial convolutions. Partial convolutions are useful in scenarios where the input data has missing or corrupted regions, as they prevent the corrupted regions from affecting the convolution operation in the valid regions.

## Usage

```python
import torch.nn as nn

class PartialConv1d(nn.Conv1d):
    def __init__(self, *args, multi_channel=False, return_mask=False, **kwargs):
        ...

    def forward(self, input, mask_in=None):
        ...
```

To create an instance of `PartialConv1d`, you can use it the same way as `nn.Conv1d`:

```python
partial_conv = PartialConv1d(in_channels, out_channels, kernel_size, multi_channel=True, return_mask=False)
```

The `forward` method takes an input tensor and an optional mask tensor (`mask_in`). If no mask is provided, it creates a new mask tensor.

```python
output = partial_conv(input_tensor)
# or
output, mask = partial_conv(input_tensor, mask_in=mask_tensor, return_mask=True)
```

## Parameters

- `*args`: Positional arguments for `nn.Conv1d` (e.g., `in_channels`, `out_channels`, `kernel_size`).
- `multi_channel` (bool, optional): Determines whether the mask is multi-channel or not. Default is `False`.
- `return_mask` (bool, optional): If set to `True`, the forward method will return the output and the updated mask. Default is `False`.
- `**kwargs`: Keyword arguments for `nn.Conv1d` (e.g., `stride`, `padding`, `dilation`).

## Functionality

1. **Initialization**: The `__init__` method initializes the class and handles optional parameters (`multi_channel` and `return_mask`).
2. **Weight Mask Updater**: The `weight_maskUpdater` attribute is a tensor used to update the mask during the forward pass.
3. **Forward Pass**: The `forward` method performs the partial convolution operation:
   - If a mask is provided or the input size has changed, it updates the mask and computes the mask ratio.
   - If no mask is provided, it creates a new mask tensor.
   - It applies the mask to the input tensor by element-wise multiplication.
   - It performs the regular convolution operation using the `nn.Conv1d` forward method.
   - It applies the mask ratio to the output and updates the output using the mask.
   - It returns the final output and, optionally, the updated mask.
4. **Mask Update**: The mask is updated during the forward pass using a convolution operation with the `weight_maskUpdater` tensor.
5. **Output Computation**: The output is computed by multiplying the raw output from the convolution with the mask ratio, applying the mask, and, if a bias is present, adding the bias.

## Example

```python
import torch
import torch.nn as nn

# Create a PartialConv1d instance
partial_conv = PartialConv1d(3, 6, 3, multi_channel=True, return_mask=True)

# Create input tensor and mask
input_tensor = torch.randn(2, 3, 10)
mask_tensor = torch.ones_like(input_tensor)
mask_tensor[:, :, 5:] = 0  # Simulate missing regions

# Perform partial convolution
output, mask = partial_conv(input_tensor, mask_in=mask_tensor, return_mask=True)
```

In this example, a `PartialConv1d` instance is created with 3 input channels, 6 output channels, and a kernel size of 3. The `multi_channel` parameter is set to `True`, and `return_mask` is set to `True`. An input tensor and a mask tensor are created, where the mask tensor has missing regions simulated by setting the values to 0. The partial convolution is then performed using the `forward` method, and the output tensor and updated mask are returned.
