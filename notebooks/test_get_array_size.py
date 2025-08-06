import torch
import xbatcher
import xarray as xr
import numpy as np
import pytest

from functions import _get_output_array_size

@pytest.fixture
def bgen_fixture() -> xbatcher.BatchGenerator:
    data = xr.DataArray(
        data=np.random.rand(100, 100, 10),
        dims=("x", "y", "t"),
        coords={
            "x": np.arange(100),
            "y": np.arange(100),
            "t": np.arange(10),
        }
    )
    
    bgen = xbatcher.BatchGenerator(
        data,
        input_dims=dict(x=10, y=10),
        input_overlap=dict(x=5, y=5),
    )
    return bgen

@pytest.mark.parametrize(
    "case_description, output_tensor_dim, new_dim, resample_dim, expected_output",
    [
        (
            "Resampling only: Downsample x, Upsample y",
            {'x': 5, 'y': 20},  
            [],
            ['x', 'y'],
            {'x': 50, 'y': 200} 
        ),
        (
            "New dimensions only: Add a 'channel' dimension",
            {'channel': 3},
            ['channel'],
            [],
            {'channel': 3}
        ),
        (
            "Mixed: Resample x and add new channel dimension",
            {'x': 30, 'channel': 12}, 
            ['channel'],
            ['x'],
            {'x': 300, 'channel': 12} 
        ),
        (
            "Identity resampling (ratio=1)",
            {'x': 10, 'y': 10},
            [],
            ['x', 'y'],
            {'x': 100, 'y': 100} 
        ),
        (
            "Dimension not in batcher is treated as new",
            {'t': 5},
            ['t'],
            [],
            {'t': 5}
        )
        
    ]
)
def test_get_output_array_size_scenarios(
    bgen_fixture,  # The fixture is passed as an argument
    case_description,
    output_tensor_dim,
    new_dim,
    resample_dim,
    expected_output
):
    """
    Tests various valid scenarios for calculating the output array size.
    The `case_description` parameter is not used in the code but helps make
    test results more readable.
    """
    # The `bgen_fixture` argument is the BatchGenerator instance created by our fixture
    result = _get_output_array_size(
        bgen=bgen_fixture,
        output_tensor_dim=output_tensor_dim,
        new_dim=new_dim,
        resample_dim=resample_dim
    )
    
    assert result == expected_output, f"Failed on case: {case_description}"

def test_get_output_array_size_raises_assertion_error_on_non_integer_size():
    """
    Tests that the function raises an AssertionError when the resampling
    calculation results in a non-integer output dimension size.
    """
    # DataArray size for 'x' is 101.
    data_for_error = xr.DataArray(
        data=np.random.rand(101, 100, 10),
        dims=("x", "y", "t")
    )
    
    bgen = xbatcher.BatchGenerator(data_for_error, input_dims={'x': 10})
    
    # The resampling logic will be: 101 * (5 / 10) = 50.5, which is not an integer.
    output_tensor_dim = {'x': 5}
    
    with pytest.raises(AssertionError):
        _get_output_array_size(
            bgen=bgen,
            output_tensor_dim=output_tensor_dim,
            new_dim=[],
            resample_dim=['x']
        )
