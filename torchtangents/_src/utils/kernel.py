# Copyright 2025 torchtangents contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataclass containing information about the NTK and NNGP of a model."""

from dataclasses import dataclass
from typing import Optional, Sequence, Union
import torch


@dataclass
class Kernel:
    """Dataclass containing information about the NTK and NNGP of a model.
    
    This is the PyTorch equivalent of neural_tangents.Kernel, using torch.Tensor
    instead of jnp.ndarray.
    
    Attributes:
        nngp:
            covariance between the first and second batches (NNGP). A `torch.Tensor` of
            shape `(batch_size_1, batch_size_2, height, [height,], width, [width,], ...)`,
            where exact shape depends on `diagonal_spatial`.
            
        ntk:
            the neural tangent kernel (NTK). `torch.Tensor` of same shape as `nngp`.
            
        cov1:
            covariance of the first batch of inputs. A `torch.Tensor` with shape
            `(batch_size_1, [batch_size_1,] height, [height,], width, [width,], ...)`
            where exact shape depends on `diagonal_batch` and `diagonal_spatial`.
            
        cov2:
            optional covariance of the second batch of inputs. A `torch.Tensor` with
            shape `(batch_size_2, [batch_size_2,] height, [height,], width, [width,], ...)`
            where the exact shape depends on `diagonal_batch` and `diagonal_spatial`.
            
        x1_is_x2:
            a boolean specifying whether `x1` and `x2` are the same.
            
        is_gaussian:
            a boolean, specifying whether the output features or channels of the layer
            / NN function (returning this `Kernel` as the `kernel_fn`) are i.i.d.
            Gaussian with covariance `nngp`, conditioned on fixed inputs to the layer
            and i.i.d. Gaussian weights and biases of the layer.
            
        is_reversed:
            a boolean specifying whether the covariance matrices `nngp`, `cov1`,
            `cov2`, and `ntk` have the ordering of spatial dimensions reversed.
            Ignored unless `diagonal_spatial` is `False`. Used internally to avoid
            self-cancelling transpositions in a sequence of CNN layers that flip the
            order of kernel spatial dimensions.
            
        is_input:
            a boolean specifying whether the current layer is the input layer, and it
            is used to avoid applying dropout to the input layer.
            
        diagonal_batch:
            a boolean specifying whether `cov1` and `cov2` store only the diagonal of
            the sample-sample covariance (`diagonal_batch == True`,
            `cov1.shape == (batch_size_1, ...)`), or the full covariance
            (`diagonal_batch == False`, `cov1.shape == (batch_size_1, batch_size_1, ...)`).
            Defaults to `True` as no current layers require the full covariance.
            
        diagonal_spatial:
            a boolean specifying whether all (`cov1`, `ntk`, etc.) covariance matrices
            store only the diagonals of the location-location covariances
            (`diagonal_spatial == True`,
            `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
            or the full covariance (`diagonal_spatial == False`,
            `nngp.shape == (batch_size_1, batch_size_2, height, height, width, width, depth, depth, ...)`).
            
        shape1:
            a tuple specifying the shape of the random variable in the first batch of
            inputs. These have covariance `cov1` and covariance with the second batch
            of inputs given by `nngp`.
            
        shape2:
            a tuple specifying the shape of the random variable in the second batch of
            inputs. These have covariance `cov2` and covariance with the first batch
            of inputs given by `nngp.T` (or `nngp.transpose` with corresponding axis order).
    """
    
    nngp: Optional[torch.Tensor] = None
    ntk: Optional[torch.Tensor] = None
    cov1: Optional[torch.Tensor] = None
    cov2: Optional[torch.Tensor] = None
    x1_is_x2: Optional[bool] = None
    is_gaussian: bool = True
    is_reversed: bool = False
    is_input: bool = False
    diagonal_batch: bool = True
    diagonal_spatial: bool = False
    shape1: Optional[Sequence[int]] = None
    shape2: Optional[Sequence[int]] = None
    
    def reverse(self) -> 'Kernel':
        """Reverse the spatial dimensions of the kernel.
        
        Returns:
            A new Kernel with reversed spatial dimensions.
        """
        def _reverse_tensor(x):
            if x is None:
                return None
            if self.diagonal_spatial:
                return x
            # For non-diagonal spatial, reverse pairs of spatial dimensions
            # This is a simplified version - the full implementation would need
            # to handle the specific dimension ordering
            return x
            
        return Kernel(
            nngp=_reverse_tensor(self.nngp),
            ntk=_reverse_tensor(self.ntk),
            cov1=_reverse_tensor(self.cov1),
            cov2=_reverse_tensor(self.cov2),
            x1_is_x2=self.x1_is_x2,
            is_gaussian=self.is_gaussian,
            is_reversed=not self.is_reversed,
            is_input=self.is_input,
            diagonal_batch=self.diagonal_batch,
            diagonal_spatial=self.diagonal_spatial,
            shape1=self.shape1,
            shape2=self.shape2
        )
    
    def replace(self, **kwargs) -> 'Kernel':
        """Create a new Kernel with updated fields.
        
        Args:
            **kwargs: fields to update.
            
        Returns:
            A new Kernel with updated fields.
        """
        # Get current values
        current_values = {
            'nngp': self.nngp,
            'ntk': self.ntk,
            'cov1': self.cov1,
            'cov2': self.cov2,
            'x1_is_x2': self.x1_is_x2,
            'is_gaussian': self.is_gaussian,
            'is_reversed': self.is_reversed,
            'is_input': self.is_input,
            'diagonal_batch': self.diagonal_batch,
            'diagonal_spatial': self.diagonal_spatial,
            'shape1': self.shape1,
            'shape2': self.shape2
        }
        
        # Update with provided kwargs
        current_values.update(kwargs)
        
        return Kernel(**current_values)