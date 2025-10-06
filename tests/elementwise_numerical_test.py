# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Native PyTorch/unittest rewrite of the original JAX/absl test.

import unittest

# Your PyTorch port of the example under test:
from examples import elementwise_numerical  # must define main(args) or main()

# PyTorch test utilities (ported earlier)
from tests import test_utils


class ElementwiseNumericalTest(test_utils.NeuralTangentsTestCase):

    def test_elementwise_numerical(self):
        # Preserve the original call signature if kept during the port.
        # If your PyTorch version has main() with no args, call main() instead.
        elementwise_numerical.main(None)


if __name__ == "__main__":
    unittest.main()
