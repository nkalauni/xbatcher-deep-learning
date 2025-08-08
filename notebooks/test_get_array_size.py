import torch
import xbatcher
import xarray as xr
import numpy as np
import pytest

from functions import _get_output_array_size, _get_resample_factor

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
    "case_description, output_tensor_dim, new_dim, core_dim, resample_dim, expected_output",
    [
        (
            "Resampling only: Downsample x, Upsample y",
            {'x': 5, 'y': 20},  
            [],
            [],
            ['x', 'y'],
            {'x': 50, 'y': 200} 
        ),
        (
            "New dimensions only: Add a 'channel' dimension",
            {'channel': 3},
            ['channel'],
            [],
            [],
            {'channel': 3}
        ),
        (
            "Mixed: Resample x, add new channel dimension and keep t as core",
            {'x': 30, 'channel': 12}, 
            ['channel'],
            ['t'],
            ['x'],
            {'x': 300, 'channel': 12} 
        ),
        (
            "Identity resampling (ratio=1)",
            {'x': 10, 'y': 10},
            [],
            [],
            ['x', 'y'],
            {'x': 100, 'y': 100} 
        ),
        (
            "Core dims only: 't' is a core dim",
            {'t': 10},
            [], 
            ['t'], 
            [],
            {'t': 10}
        ),
    ]
)
def test_get_output_array_size_scenarios(
    bgen_fixture,  # The fixture is passed as an argument
    case_description,
    output_tensor_dim,
    new_dim,
    core_dim,
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
        core_dim=core_dim,
        resample_dim=resample_dim
    )

    assert result == expected_output, f"Failed on case: {case_description}"

def test_get_output_array_size_raises_error_on_mismatched_core_dim(bgen_fixture):
    """Tests ValueError when a core_dim size doesn't match the source."""
    with pytest.raises(ValueError, match="does not equal the source data array size"):
        _get_output_array_size(
            bgen_fixture, output_tensor_dim={'t': 99}, new_dim=[], core_dim=['t'], resample_dim=[]
        )

def test_get_output_array_size_raises_error_on_unspecified_dim(bgen_fixture):
    """Tests ValueError when a dimension is not specified in any category."""
    with pytest.raises(ValueError, match="must be specified in one of"):
        _get_output_array_size(
            bgen_fixture, output_tensor_dim={'x': 10}, new_dim=[], core_dim=[], resample_dim=[]
        )

def test_get_resample_factor_raises_error_on_invalid_ratio(bgen_fixture):
    """Tests AssertionError when the resample ratio is not an integer or its inverse."""
    with pytest.raises(AssertionError, match="must be an integer or its inverse"):
        # 15 / 10 = 1.5, which is not a valid ratio
        _get_resample_factor(bgen_fixture, output_tensor_dim={'x': 15}, resample_dim=['x'])
