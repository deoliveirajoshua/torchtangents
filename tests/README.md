# TorchTangents Test Structure

This directory contains the reorganized test suite for TorchTangents, split into two main categories:

## Directory Structure

```
tests/
├── run_tests.py                 # Main test runner
├── torchtangents_tests/         # TorchTangents-specific tests
│   ├── __init__.py
│   ├── test_basic.py           # Basic functionality tests
│   ├── test_batching.py        # Batching tests
│   ├── test_empirical.py       # Empirical kernel tests
│   ├── test_monte_carlo.py     # Monte Carlo tests
│   ├── test_predict.py         # Prediction tests
│   ├── test_stax_combinators.py # Stax combinator tests
│   ├── test_stax_elementwise.py # Stax elementwise tests
│   ├── test_stax_linear.py     # Stax linear layer tests
│   ├── test_structured_derivatives.py # Structured derivatives tests
│   └── test_utils.py           # TorchTangents test utilities
└── neural_tangents_tests/       # Neural-Tangents compatibility tests
    ├── __init__.py
    ├── test_utils.py           # PyTorch-adapted neural-tangents test utilities
    ├── empirical_test.py       # Empirical kernel compatibility tests
    ├── stax/                   # Stax layer tests
    │   ├── __init__.py
    │   └── stax_test.py        # Basic stax layer tests
    └── experimental/           # Experimental feature tests
        └── __init__.py
```

## Test Categories

### TorchTangents Tests (`torchtangents_tests/`)

These tests focus on the PyTorch-specific implementation and functionality:

- **Core functionality**: Basic PyTorch integration and tensor operations
- **Performance**: PyTorch-specific optimizations and GPU support
- **API compatibility**: Ensuring the PyTorch API matches expected patterns
- **New features**: TorchTangents-specific extensions (like structured derivatives)

### Neural-Tangents Compatibility Tests (`neural_tangents_tests/`)

These tests are adapted from the original neural-tangents JAX library to ensure:

- **Mathematical correctness**: Results match neural-tangents where possible
- **API compatibility**: Function signatures and behavior match neural-tangents
- **Reference implementation**: Provide ground truth for correctness verification
- **Regression testing**: Catch compatibility breaks with neural-tangents

## Running Tests

### Run All Tests
```bash
cd tests
python run_tests.py --suite all
```

### Run Only TorchTangents Tests
```bash
cd tests  
python run_tests.py --suite torchtangents
```

### Run Only Compatibility Tests
```bash
cd tests
python run_tests.py --suite compatibility
```

### Run Individual Test Files
```bash
# TorchTangents-specific test
python -m pytest torchtangents_tests/test_structured_derivatives.py

# Neural-tangents compatibility test  
python -m pytest neural_tangents_tests/empirical_test.py

# Stax layer tests
python -m pytest neural_tangents_tests/stax/stax_test.py
```

## Adding New Tests

### For TorchTangents-specific functionality:
1. Add tests to `torchtangents_tests/`
2. Use standard PyTorch testing patterns
3. Focus on PyTorch-specific behavior and performance

### For Neural-Tangents compatibility:
1. Add tests to `neural_tangents_tests/`
2. Adapt from corresponding neural-tangents test files
3. Use the `TorchTangentsTestCase` base class for consistency
4. Focus on mathematical correctness and API compatibility

## Test Utilities

- `torchtangents_tests/test_utils.py`: Standard PyTorch testing utilities
- `neural_tangents_tests/test_utils.py`: Adapted neural-tangents testing framework for PyTorch

## Dependencies

The compatibility tests may require additional dependencies:
- `absl-py` for parameterized testing (from neural-tangents style)
- Standard PyTorch testing dependencies

## Future Extensions

This structure allows for easy addition of:
- Performance benchmarks
- Integration tests with other libraries
- Hardware-specific tests (GPU, TPU when available)
- More comprehensive neural-tangents test coverage