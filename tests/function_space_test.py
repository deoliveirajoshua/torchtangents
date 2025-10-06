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

from examples import function_space   # PyTorch-ported module exposing main
from tests import test_utils          # PyTorch NeuralTangentsTestCase utilities


class FunctionSpaceTest(test_utils.NeuralTangentsTestCase):

    def test_function_space(self):
        # Keep parity with the original signature; change to function_space.main()
        # if your PyTorch port takes no arguments.
        function_space.main(None)


if __name__ == "__main__":
    unittest.main()
