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

from examples import elementwise  # PyTorch-ported module with main
from tests import test_utils      # PyTorch version of NeuralTangentsTestCase

class ElementwiseTest(test_utils.NeuralTangentsTestCase):

    def test_elementwise(self):
        # Preserve original call style; switch to elementwise.main() if no args.
        elementwise.main()

if __name__ == "__main__":
    unittest.main()
