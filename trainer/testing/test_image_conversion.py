import unittest
import easyocr

class MyTestCase(unittest.TestCase):

    def test_download(self):
        reader = easyocr.Reader(['en'],  model_storage_directory='../saved_models')
        result = reader.readtext('./unittest_data/0.png')
        print(f"{result=}")

    def test_image(self):
        # reader = easyocr.Reader(['en'],
        #                         model_storage_directory='../saved_models',
        #                         # recog_network='generation1',
        #                         # user_network_directory='../saved_models',
        #                         download_enabled=False
        #                         )
        reader = easyocr.Reader(['en'],
                                recog_network='best_accuracy',
                                model_storage_directory='../saved_models',
                                user_network_directory='../saved_models',
                                download_enabled=False,
                                verbose=True
                                )
        result = reader.readtext('./unittest_data/4.png',detail=0)
        print(f"{result=}")



if __name__ == '__main__':
    unittest.main()
