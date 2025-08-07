import xarray as xr
import numpy as np
import xbatcher
from xbatcher.loaders.torch import MapDataset
import torch

def _get_resample_factor(
    bgen: xbatcher.BatchGenerator,
    output_tensor_dim: dict[str, int],
    resample_dim: list[str]
):
    resample_factor = {}
    for dim in resample_dim:
        r = output_tensor_dim[dim] / bgen.input_dims[dim]
        assert r.is_integer() or (r ** -1).is_integer()
        resample_factor[dim] = output_tensor_dim[dim] / bgen.input_dims[dim]

    return resample_factor

def _get_output_array_size(
    bgen: xbatcher.BatchGenerator,
    output_tensor_dim: dict[str, int],
    new_dim: list[str],
    resample_dim: list[str]
):
    resample_factor = _get_resample_factor(bgen, output_tensor_dim, resample_dim)
    output_size = {}
    for key, size in output_tensor_dim.items():
        if key in new_dim:
            # This is a new axis, size is determined
            # by the tensor size.
            output_size[key] = output_tensor_dim[key]
        else:
            # This is a resampled axis, determine the new size
            # by the ratio of the batchgen window to the tensor size.
            temp_output_size = bgen.ds.sizes[key] * resample_factor[key]
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
    # TODO input checking

    # Get resample factors
    resample_factor = _get_resample_factor(dataset.X_generator, output_tensor_dim, resample_dim)
    
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
            # Perhaps use xbatcher _gen_slices here?
            new_indexer = {}
            for key in old_indexer:
                if key in resample_dim:
                    new_indexer[key] = slice(
                        int(old_indexer[key].start * resample_factor[key]),
                        int(old_indexer[key].stop * resample_factor[key])
                    )
            
            output_da.loc[new_indexer] += out_batch[ib, ...]
            output_n.loc[new_indexer] += 1
    
    output_da = output_da / output_n

    return output_da