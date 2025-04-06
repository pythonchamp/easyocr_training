import unittest
import easyocr

class MyTestCase(unittest.TestCase):

    def test_download(self):
        reader = easyocr.Reader(['en'],  model_storage_directory='../saved_models')
        result = reader.readtext('./unittest_data/0.png')
        print(f"{result=}")


    def test_custom_image(self):
        reader = easyocr.Reader(['en'], recog_network='custom',
                                model_storage_directory='../saved_models')
        result = reader.readtext('./unittest_data/0.png')
        print(f"{result=}")


if __name__ == '__main__':
    unittest.main()
