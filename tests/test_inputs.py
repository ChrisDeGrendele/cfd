import pytest
from unittest.mock import patch
from configparser import NoSectionError, NoOptionError
import os
import sys

# sys.path.append(os.path.abspath('../1D'))
# from inputs import Inputs
from onedim import Inputs

INI_FILE_PATH = 'tests/inputs.ini'

# Basic test for verifying correct inputs handling
def test_correct_inputs():
    inputs = Inputs(INI_FILE_PATH)
    assert inputs.nx == 100
    assert inputs.numghosts == 3
    assert inputs.xlim == (0, 1)
    assert inputs.time_steps == 100
    assert inputs.method == 'RK1'
    assert inputs.t0 == 0
    assert inputs.t_finish == 0.1
    assert inputs.ics == 'sod'
    assert inputs.flux == 'weno5'
    assert inputs.bc_lo == 'zerograd'
    assert inputs.bc_hi == 'zerograd'
    assert inputs.output_freq == 1
    assert inputs.make_movie is True

#Mock test for missing mandatory option
def test_missing_mandatory_option():
    with patch('configparser.ConfigParser.get', side_effect=NoOptionError('nx', 'Mesh')) as mock_get:
        with pytest.raises(ValueError) as excinfo:
            inputs = Inputs(INI_FILE_PATH)
        assert "Missing mandatory argument: 'nx' in section 'Mesh'" in str(excinfo.value)
        mock_get.assert_called_once_with('Mesh', 'nx', fallback=None)



# # Mock test for type conversion error
def test_type_conversion_error():
    with patch('configparser.ConfigParser.get', return_value='not_a_number') as mock_get:
        with pytest.raises(ValueError) as excinfo:
            inputs = Inputs(INI_FILE_PATH)
        assert "Type conversion error for" in str(excinfo.value)
        mock_get.assert_called()



