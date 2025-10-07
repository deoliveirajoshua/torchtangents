# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# This is a native PyTorch/unittest rewrite of the original JAX/absl test.

import unittest

# Use your PyTorch port:
# from examples import weight_space  # if you kept the same module name
from examples import weight_space  # assumes a PyTorch-ported module with main(args)

# PyTorch version of the utilities from the previous message
from tests import test_utils


class WeightSpaceTest(test_utils.NeuralTangentsTestCase):

    def test_weight_space(self):
        # Keep signature parity with the original: main(None)
        # (If your PyTorch main() takes no args, just call weight_space.main().)
        weight_space.main()


if __name__ == "__main__":
    unittest.main()
