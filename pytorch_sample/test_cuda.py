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
        print("Is CUDA available?", status)
        print("Number of GPUs:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(0))
        print("Device name:", torch.version.cuda)
        # Assert that CUDA is available
        assert status == True


if __name__ == '__main__':
    unittest.main()
