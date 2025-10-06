# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Native PyTorch/unittest rewrite of the original JAX/absl test.

import unittest

from examples import empirical_ntk   # PyTorch-ported module with main
from tests import test_utils         # PyTorch version of NeuralTangentsTestCase


class EmpiricalNtkTest(test_utils.NeuralTangentsTestCase):

    def test_empirical_ntk(self):
        # Preserve original call style; use empirical_ntk.main() if no args.
        empirical_ntk.main(None)


if __name__ == "__main__":
    unittest.main()
