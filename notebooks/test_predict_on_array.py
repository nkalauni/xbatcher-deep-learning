import xarray as xr
import numpy as np
import torch
import xbatcher
import pytest
from xbatcher.loaders.torch import MapDataset

from functions import _get_output_array_size, predict_on_array
from dummy_models import *

@pytest.fixture
def map_dataset_fixture() -> MapDataset:
    """
    Creates a MapDataset with a predictable BatchGenerator for testing.
    - Data is an xarray DataArray with dimensions x=20, y=10
    - Values are a simple np.arange sequence for easy verification.
    - Batches are size x=10, y=5 with overlap x=2, y=2
    """
    # Using a smaller, more manageable dataset for testing
    data = xr.DataArray(
        data=np.arange(20 * 10).reshape(20, 10),
        dims=("x", "y"),
        coords={"x": np.arange(20), "y": np.arange(10)}
    ).astype(float)
    
    bgen = xbatcher.BatchGenerator(
        data,
        input_dims=dict(x=10, y=5),
        input_overlap=dict(x=2, y=2),
    )
    return MapDataset(bgen)
