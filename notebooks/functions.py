import xarray as xr
import numpy as np
import xbatcher
from xbatcher.loaders.torch import MapDataset
import torch

def _get_output_array_size(
    bgen: xbatcher.BatchGenerator,
    output_tensor_dim: dict[str, int],
    new_dim: list[str],
    resample_dim: list[str]
):
    output_size = {}
    for key, size in output_tensor_dim.items():
        if key in new_dim:
            # This is a new axis, size is determined
            # by the tensor size.
            output_size[key] = output_tensor_dim[key]
        else:
            # This is a resampled axis, determine the new size
            # by the ratio of the batchgen window to the tensor size.
            window_size = bgen.input_dims[key]
            tensor_size = output_tensor_dim[key]
            resample_ratio = tensor_size / window_size
    
            temp_output_size = bgen.ds.sizes[key] * resample_ratio
            assert temp_output_size.is_integer()
            output_size[key] = int(temp_output_size)
    return output_size

def predict_on_array(
    dataset: MapDataset,
    model: torch.nn.Module,
    output_tensor_dim: dict[str, int],
    new_dim: list[str],
    resample_dim: list[str],
    batch_size: int=16
):
    # Set up output array
    output_size = _get_output_array_size(dataset.X_generator, output_tensor_dim, new_dim, resample_dim)
            
    output_da = xr.DataArray(
        data=np.zeros(tuple(output_size.values())),
        dims=tuple(output_size.keys())
    )
    output_n = xr.full_like(output_da, 0)
    
    # Prepare data laoder
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Iterate over each batch
    for i, batch in enumerate(loader):
        out_batch = model(batch).detach().numpy()

        # Iterate over each example in the batch
        for ib in range(out_batch.shape[0]):
            # Get the slice object associated with this example
            old_indexer = dataset.X_generator._batch_selectors.selectors[(i*batch_size)+ib][0]
            # Only index into axes that are resampled, rescaling the bounds
            new_indexer = dict()
            for key in old_indexer:
                if key in resample_dim:
                    resample_ratio = output_tensor_dim[key] / dataset.X_generator.input_dims[key]
                    new_indexer[key] = slice(
                        int(old_indexer[key].start * resample_ratio),
                        int(old_indexer[key].stop * resample_ratio)
                    )
            
            output_da.loc[new_indexer] += out_batch[ib, ...]
            output_n.loc[new_indexer] += 1
    
    # TODO aggregate output
    return output_da