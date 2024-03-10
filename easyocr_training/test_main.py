import unittest
import easyocr

from common.perf_counter import PerfCounter
from easyocr_training.main import Reader


class MyTestCase(unittest.TestCase):

    @PerfCounter.calculate_execution_time
    def test_converting_image_to_string(self):
        image_path = "./images/testimages/en_letters.png"
        reader = Reader()
        expected = ["ABCDEFGHIJKLMNOP", "abcdefghijklmnop"]
        actual = reader.readtext(image_path)
        print(f"{actual=}")
        self.assertEqual(actual, expected)  # add assertion here


if __name__ == '__main__':
    unittest.main()
