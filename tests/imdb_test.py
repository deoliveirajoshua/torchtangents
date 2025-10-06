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

from examples import imdb          # PyTorch-ported module exposing main()
from tests import test_utils       # PyTorch version of NeuralTangentsTestCase


class ImdbTest(test_utils.NeuralTangentsTestCase):

    def test_imdb(self):
        # Keep API parity with the original test by using dummy data.
        imdb.main(use_dummy_data=True)


if __name__ == "__main__":
    unittest.main()
