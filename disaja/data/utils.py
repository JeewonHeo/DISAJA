from __future__ import annotations
import dataclasses
import numpy as np
import torch
from torch import Tensor

@dataclasses.dataclass
class TensorCollection:

    def keys(self) -> list[str]:
        return [field.name for field in dataclasses.fields(self)]

    def to(self, device):
        def convert(data):
            if torch.is_tensor(data):
                data = data.to(device)
            return data
        batch = [convert(each) for each in dataclasses.astuple(self)]
        return self.__class__(*batch)

    def cpu(self):
        return self.to(torch.device('cpu'))

    # TODO
    def numpy(self):
        raise NotImplementedError

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        output = [f'{self.__class__.__name__}:']
        for key, value in dataclasses.asdict(self).items():
            if torch.is_tensor(value):
                shape = tuple(value.shape)
                dtype = value.dtype
                device = value.device.type
                output.append(f'- {key}: {shape=}, {dtype=}, {device=}')
            elif isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                output.append(f'- {key}: type=np.ndarray, {shape=}, {dtype=}')
            elif isinstance(value, list):
                output.append(f'- {key}: type=list[{type(value[0]).__name__}]')
            else:
                output.append(f'- {key}: type={type(value).__name__}')
        output = '\n'.join(output)
        return output

    def __len__(self) -> int:
        r"""docstring TODO
        warning: batch_first
        """
        sizes = {len(each) for each in dataclasses.astuple(self)}
        if len(sizes) != 1:
            raise RuntimeError
        return sizes.pop()



@torch.jit.script # type: ignore
def make_data_mask(data: Tensor, length: Tensor) -> Tensor:
    """docstring for make_data_mask TODO
    """
    mask_shape = data.shape[:-1]
    mask = torch.full(size=mask_shape, fill_value=0, dtype=torch.bool,
                      device=data.device)
    for m, l in zip(mask, length):
        m[: l].fill_(1)
    return mask
