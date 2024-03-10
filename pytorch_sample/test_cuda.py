import unittest
import torch


class MyTestCase(unittest.TestCase):
    """
    This function tests something and does not have any parameters or return types.
    """

    def test_something(self):
        """
        Test function to check the availability of CUDA and print a tensor.
        """
        # Create a 5x3 tensor of random numbers
        x = torch.rand(5, 3)
        print(x)

        # Check the availability of CUDA
        status = torch.cuda.is_available()

        # Assert that CUDA is available
        assert status == True


if __name__ == '__main__':
    unittest.main()
