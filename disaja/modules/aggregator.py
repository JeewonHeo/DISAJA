import torch
from torch import Tensor
import torch.nn as nn

@torch.jit.script
def make_scatter_add_index(lengths: Tensor,
                           num_features: int,
) -> Tensor:
    r"""
    Example::
      >>> make_scatter_add_index(lengths=torch.tensor([2, 1, 3]))
      torch.tensor([0, 0, 1, 2, 2, 2])
    """
    index = [lengths.new_full(size=[int(each), ], fill_value=idx) for idx, each in enumerate(lengths)]
    index = torch.cat(index)
    index = index.unsqueeze(1).repeat(1, num_features)
    return index

class ScatterMean(nn.Module):
    def forward(self,
                input: Tensor,
                data_mask: Tensor,
                lengths: Tensor) -> Tensor:
        """Computes the mean of input values per segment using scatter operation.
        
        Args:
            input (Tensor): The input tensor of shape (batch_size, seq_len, num_features).
            data_mask (Tensor): A boolean mask indicating valid input positions.
            lengths (Tensor): A tensor indicating the number of valid elements per batch.
        
        Returns:
            Tensor: The mean of input values per segment of shape (batch_size, num_features).
        """
        batch_size, _, num_features = input.shape

        # Expand data_mask for broadcasting
        data_mask = data_mask.unsqueeze(2)

        # Select valid input values based on data_mask
        input = input.masked_select(data_mask)
        input = input.reshape(-1, num_features)

        # Compute index mapping for scatter operation
        index = self.make_scatter_add_index(lengths, num_features)

        # Perform scatter_add operation to sum values in each segment
        output = torch.scatter_add(
            input=input.new_zeros((batch_size, num_features)),
            dim=0,
            index=index,
            src=input)

        # Compute mean by dividing summed values by segment lengths
        output = output / lengths.unsqueeze(1).to(output.dtype)
        return output


    @staticmethod
    def make_scatter_add_index(lengths: Tensor,
                               num_features: int,
    ) -> Tensor:
        """Static method to generate scatter add index
        """
        return make_scatter_add_index(lengths=lengths,
                                      num_features=num_features)

