"""Test utilities adapted from neural-tangents for PyTorch.

This module provides testing utilities that mirror the neural-tangents test infrastructure
but adapted for PyTorch tensors and torchtangents functionality.
"""

import dataclasses
import itertools
import logging
import os
from types import ModuleType
from typing import Callable, Optional, Sequence, Union, Dict, Any

import torch
import numpy as np
from absl import flags
from absl.testing import parameterized


flags.DEFINE_string(
    'tt_test_device',
    '',
    help='Device to run tests on (cpu, cuda, mps)'
)

flags.DEFINE_integer(
    'tt_num_generated_cases',
    int(os.getenv('TT_NUM_GENERATED_CASES', '4')),
    help='Number of generated cases to test'
)

FLAGS = flags.FLAGS


def device_under_test() -> str:
    """Get the device being used for testing."""
    try:
        device = getattr(FLAGS, 'tt_test_device', None)
        if device:
            return device
    except:
        # Flags not parsed yet, fall back to auto-detection
        pass
    
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


_DEFAULT_TOLERANCE: Dict[torch.dtype, float] = {
    torch.bool: 0,
    torch.int32: 0,
    torch.int64: 0,
    torch.float16: 5e-3,
    torch.float32: 5e-3,
    torch.float64: 1e-5,
    torch.bfloat16: 5e-3
}


def _default_tolerance() -> Dict[torch.dtype, float]:
    """Get default tolerance for device under test."""
    device = device_under_test()
    if device == 'cuda':
        # CUDA often has slightly different numerical behavior
        tol = _DEFAULT_TOLERANCE.copy()
        tol[torch.float32] = 1e-2
        return tol
    return _DEFAULT_TOLERANCE


def _tolerance(dtype: torch.dtype, tol: Optional[float] = None) -> float:
    """Get tolerance for given dtype."""
    if tol is not None and not isinstance(tol, dict):
        return tol
    
    tol_dict = tol if isinstance(tol, dict) else {}
    return tol_dict.get(dtype, _default_tolerance()[dtype])


def _assert_torch_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    err_msg: str = ''
):
    """Assert that two PyTorch tensors are close."""
    if atol is None:
        atol = _tolerance(a.dtype)
    if rtol is None:
        rtol = _tolerance(a.dtype)
    
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=err_msg)
    except AssertionError as e:
        raise AssertionError(f"{err_msg}\n{str(e)}")


def _is_sequence(x) -> bool:
    """Check if x is a sequence."""
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


_CACHED_INDICES: Dict[int, Sequence[int]] = {}


def _cases_from_list(xs):
    """Sample test cases from a list."""
    xs = list(xs)
    n = len(xs)
    try:
        max_cases = FLAGS.tt_num_generated_cases
    except:
        # Default if flags not parsed
        max_cases = 10
        
    if n < max_cases:
        return xs
    k = min(n, max_cases)
    # Random sampling for every parameterized test is expensive. Do it once and
    # cache the result.
    indices = _CACHED_INDICES.get(n)
    if indices is None:
        rng = np.random.RandomState(42)
        _CACHED_INDICES[n] = indices = rng.permutation(n)
    return [xs[i] for i in indices[:k]]


def product(*kwargs_seqs, **testgrid):
    """Test case decorator to randomly subset a cartesian product of parameters."""
    for name, values in testgrid.items():
        assert isinstance(values, (list, tuple)), (
            'Values of {} must be given as list or tuple, found {}'.format(
                name, type(values)))

    prior_arg_names = set()
    for kwargs_seq in kwargs_seqs:
        assert ((isinstance(kwargs_seq, (list, tuple))) and
                all(isinstance(kwargs, dict) for kwargs in kwargs_seq)), (
                    'Positional parameters must be a sequence of keyword arg'
                    'dicts, found {}'
                    .format(kwargs_seq))
        if kwargs_seq:
            arg_names = set(kwargs_seq[0])
            assert all(set(kwargs) == arg_names for kwargs in kwargs_seq), (
                'Keyword argument dicts within a single parameter must all have the '
                'same keys, found {}'.format(kwargs_seq))
            assert not (arg_names & prior_arg_names), (
                'Keyword argument dict sequences must all have distinct argument '
                'names, found duplicate(s) {}'
                .format(sorted(arg_names & prior_arg_names)))
            prior_arg_names |= arg_names

    assert not (prior_arg_names & set(testgrid)), (
        'Arguments supplied in kwargs dicts in positional parameters must not '
        'overlap with arguments supplied as named parameters; found duplicate '
        'argument(s) {}'.format(sorted(prior_arg_names & set(testgrid))))

    testgrid = (tuple({k: v} for v in vs) for k, vs in testgrid.items())
    testgrid = tuple(kwargs_seqs) + tuple(testgrid)

    testcases = [
        dict(itertools.chain.from_iterable(case.items()
                                           for case in cases))
        for cases in itertools.product(*testgrid)
    ]
    return parameters(testcases)


def parameters(testcases):
    """A decorator for parameterized tests randomly sampled from the list."""
    subset = _cases_from_list(testcases)
    return parameterized.parameters(subset)


class TorchTangentsTestCase(parameterized.TestCase):
    """Testing helper class for TorchTangents tests."""

    def _assertAllClose(
        self,
        x,
        y,
        *,
        check_dtypes: bool = True,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        err_msg: str = ''
    ):
        """Assert that x and y, either tensors or nested tuples/lists, are close."""
        if isinstance(x, dict):
            self.assertIsInstance(y, dict)
            self.assertEqual(set(x.keys()), set(y.keys()))
            for k in x.keys():
                self._assertAllClose(x[k], y[k], check_dtypes=check_dtypes, 
                                   atol=atol, rtol=rtol, err_msg=err_msg)
        elif _is_sequence(x) and not isinstance(x, torch.Tensor):
            self.assertTrue(_is_sequence(y) and not isinstance(y, torch.Tensor))
            self.assertEqual(len(x), len(y))
            for x_elt, y_elt in zip(x, y):
                self._assertAllClose(x_elt, y_elt, check_dtypes=check_dtypes, 
                                   atol=atol, rtol=rtol, err_msg=err_msg)
        elif isinstance(x, (torch.Tensor, np.ndarray)) or np.isscalar(x):
            self.assertTrue(isinstance(y, (torch.Tensor, np.ndarray)) or np.isscalar(y))
            if check_dtypes:
                self.assertDtypesMatch(x, y)
            self.assertTensorsAllClose(x, y, check_dtypes=False, atol=atol, 
                                     rtol=rtol, err_msg=err_msg)
        elif x == y:
            return
        else:
            raise TypeError((type(x), type(y)))

    def assertTensorsAllClose(
        self,
        x,
        y,
        *,
        check_dtypes: bool = True,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        err_msg: str = ''
    ):
        """Assert that x and y tensors are close (up to numerical tolerances)."""
        # Convert to tensors if needed
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        
        self.assertEqual(x.shape, y.shape)
        
        if atol is None:
            atol = max(_tolerance(x.dtype), _tolerance(y.dtype))
        if rtol is None:
            rtol = max(_tolerance(x.dtype), _tolerance(y.dtype))
        
        _assert_torch_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

        if check_dtypes:
            self.assertDtypesMatch(x, y)

    def assertDtypesMatch(self, x, y):
        """Assert that dtypes match."""
        if isinstance(x, torch.Tensor):
            x_dtype = x.dtype
        else:
            x_dtype = torch.as_tensor(x).dtype
            
        if isinstance(y, torch.Tensor):
            y_dtype = y.dtype
        else:
            y_dtype = torch.as_tensor(y).dtype
            
        self.assertEqual(x_dtype, y_dtype)

    def assertAllClose(
        self,
        x,
        y,
        *,
        check_dtypes: bool = True,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        check_finite: bool = True,
        err_msg: str = ''
    ):
        """Assert that x and y are close."""
        if check_finite:
            def is_finite(tensor):
                if isinstance(tensor, torch.Tensor):
                    self.assertTrue(torch.all(torch.isfinite(tensor)))
                
            self._tree_map(is_finite, x)
            self._tree_map(is_finite, y)

        def assert_close(x, y):
            self._assertAllClose(
                x, y,
                check_dtypes=check_dtypes,
                atol=atol,
                rtol=rtol,
                err_msg=err_msg,
            )

        if dataclasses.is_dataclass(x):
            self.assertIs(type(y), type(x))
            for field in dataclasses.fields(x):
                key = field.name
                x_value, y_value = getattr(x, key), getattr(y, key)
                assert_close(x_value, y_value)
        else:
            assert_close(x, y)

    def _tree_map(self, func, tree):
        """Simple tree map implementation."""
        if isinstance(tree, dict):
            return {k: self._tree_map(func, v) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            return type(tree)(self._tree_map(func, item) for item in tree)
        else:
            return func(tree)


def update_test_tolerance(f32_tol: float = 5e-3, f64_tol: float = 1e-5):
    """Update default test tolerances."""
    _DEFAULT_TOLERANCE[torch.float32] = f32_tol
    _DEFAULT_TOLERANCE[torch.float64] = f64_tol


def assert_close_matrices(self, expected, actual, rtol, atol=0.1):
    """Assert that matrices are close with detailed error reporting."""
    def assert_close(expected, actual):
        if not isinstance(expected, torch.Tensor):
            expected = torch.as_tensor(expected)
        if not isinstance(actual, torch.Tensor):
            actual = torch.as_tensor(actual)
            
        self.assertEqual(expected.shape, actual.shape)
        
        relative_error = (
            torch.linalg.norm(actual - expected) /
            torch.maximum(torch.linalg.norm(expected), torch.tensor(1e-12)))
        
        absolute_error = torch.mean(torch.abs(actual - expected))

        if (torch.isnan(relative_error) or
            relative_error > rtol or
            absolute_error > atol):
            _log(relative_error.item(), absolute_error.item(), expected, actual, False)
            self.fail(f'Relative ERROR: {relative_error.item()}, '
                     f'Absolute ERROR: {absolute_error.item()}')
        else:
            _log(relative_error.item(), absolute_error.item(), expected, actual, True)

    self._tree_map(assert_close, expected, actual)


def _log(relative_error: float, absolute_error: float, expected, actual, did_pass: bool):
    """Log test results."""
    msg = 'PASSED' if did_pass else 'FAILED'
    logging.info(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n'
                f'\n{msg} with {relative_error} relative error \n'
                f'and {absolute_error} absolute error: \n'
                f'---------------------------------------------\n'
                f'EXPECTED: \n'
                f'{expected}\n'
                f'---------------------------------------------\n'
                f'ACTUAL: \n'
                f'{actual}\n'
                f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')


def skip_test(
    self,
    msg: str = 'Skipping large tests for speed.',
    devices: tuple[str, ...] = ('cpu',)
):
    """Skip test on certain devices."""
    if device_under_test() in devices:
        raise parameterized.TestCase.skipTest(self, msg)


def mask(
    x: torch.Tensor,
    mask_constant: Optional[float],
    mask_axis: Sequence[int],
    p: float,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Apply random masking to tensor."""
    if mask_constant is not None:
        mask_shape = [1 if i in mask_axis else s
                      for i, s in enumerate(x.shape)]
        mask_mat = torch.bernoulli(torch.full(mask_shape, p), generator=generator)
        x = torch.where(mask_mat.bool(), mask_constant, x)
        x = torch.sort(x, dim=1)[0]
    return x