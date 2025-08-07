import xarray as xr
import numpy as np
import xbatcher
from xbatcher.loaders.torch import MapDataset, IterableDataset
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
    core_dim: list[str],
    resample_dim: list[str]
):
    resample_factor = _get_resample_factor(bgen, output_tensor_dim, resample_dim)
    output_size = {}
    for key, size in output_tensor_dim.items():
        if key in new_dim:
            # This is a new axis, size is determined
            # by the tensor size.
            output_size[key] = output_tensor_dim[key]
        elif key in core_dim:
            # This is an old but unmodified axis, size is
            # determined by the source array
            if output_tensor_dim[key] != bgen.ds.sizes[key]:
                raise ValueError(
                    f"Axis {key} is a core dim, but the tensor size"
                    f"({output_tensor_dim[key]}) does not equal the"
                    f"source data array size ({bgen.ds.sizes[key]})."
                )
            output_size[key] = bgen.ds.sizes[key]
        elif key in resample_dim:
            # This is a resampled axis, determine the new size
            # by the resample factor.
            temp_output_size = bgen.ds.sizes[key] * resample_factor[key]
            assert temp_output_size.is_integer()
            output_size[key] = int(temp_output_size)
        else:
            raise ValueError(f"Axis {key} must be specified in one of new_dim, core_dim, or resample_dim") 
    return output_size

def predict_on_array(
    dataset: MapDataset | IterableDataset,
    model: torch.nn.Module,
    output_tensor_dim: dict[str, int],
    new_dim: list[str],
    core_dim: list[str],
    resample_dim: list[str],
    batch_size: int=16
) -> xr.DataArray:
    '''
    Generate predictions from a PyTorch model and reassemble predictions
    into a ``xr.DataArray``, accounting for changes in dimensions. This function
    is analagous to ``xr.apply_ufunc``, except we operate on array patches instead
    of array slices and the user function is replaced with a PyTorch module.

    ``model`` is allowed to drop axes, add axes, or resize axes compared to
    the input tensor. As in ``xr.apply_ufunc``, users must specify how axes
    change via ``new_dim`` and ``resample_dim``. At least one output
    axis must be an input dimension in the ``BatchGenerator`` used to
    generate array patches.

    

    Parameters
    ----------
    ``dataset`` (``MapDataset | IterableDataset``): A dataset that uses a
    ``BatchGenerator`` to produce examples.

    ``model`` (``torch.nn.Module``): A PyTorch module that returns a single
    output tensor.

    ``output_tensor_dim`` (``dict[str, int]``): A dictionary representing the
    names and sizes of output tensor dimensions.

    ``new_dim`` (``list[str]``): A list of axes in ``output_tensor_dim`` that
    are independent of the batch generator and source array. 
    These will be inserted as new dimensions in the output ``xr.DataArray``.

    ``core_dim`` (``list[str]``): A list of axes in ``output_tensor_dim`` that
    are unchanged from the source array and not used for windowing in ``dataset``.

    ``resample_dim`` (``list[str]``): A list of axes in ``output_tensor_dim`` that
    are used to generate patches in ``dataset``. Only these axes are allowed to change
    size.

    Notes
    -----
    The output array size is determined by the axes in ``output_tensor_dim`` according
    to the below table.

    | Axis          | Output size                                     |
    |---------------|-------------------------------------------------|
    | New axis      | Same as tensor size                             |
    | Core axis     | Same as source ``xr.DataArray`` size            |
    | Resample axis | Source array size * (tensor size / window size) |

    For example, consider a super-resolution workflow where ``dataset`` generates
    patches of size (x=10, y=10), ``model`` generates patches of size (x=20, y=20),
    and the source ``xr.DataArray`` has size (x=512, y=512). From the above table,
    we see that the tensor increases the size of both axes by a factor of 2. Therefore,
    the output data array will have size (x=1024, y=2024).

    Models may coarsen or densify tensors, but must do so by an integer factor.
    '''
    # TODO input checking

    # Get resample factors
    resample_factor = _get_resample_factor(dataset.X_generator, output_tensor_dim, resample_dim)
    
    # Set up output array
    output_size = _get_output_array_size(dataset.X_generator, output_tensor_dim, new_dim, core_dim, resample_dim)
            
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