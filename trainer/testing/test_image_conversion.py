import unittest
import easyocr
from PIL import Image
class MyTestCase(unittest.TestCase):

    def test_download(self):
        reader = easyocr.Reader(['en'],  model_storage_directory='../saved_models')
        result = reader.readtext('./unittest_data/0.png')
        print(f"{result=}")

    def preprocess_image(self,image, imgH=600, imgW=600):
        image = image.resize((imgW, imgH))
        return image

    def resize_with_padding(self,image, target_w=600, target_h=64):
        w, h = image.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)

        padded = Image.new('L', (target_w, target_h), 255)
        padded.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return padded

    def test_image(self):
        image_path = './unittest_data/4.png'
        image = Image.open(image_path).convert('L')  # Grayscale
        resize_img = self.preprocess_image(image)
        resize_path = './unittest_data/resize.png'
        resize_img.save(resize_path)

        reader = easyocr.Reader(['en'],
                                recog_network='best_accuracy',
                                model_storage_directory='../saved_models',
                                user_network_directory='../saved_models',
                                download_enabled=False,
                                verbose=True
                                )
        result = reader.readtext(resize_path)
        print(f"{result=}")



if __name__ == '__main__':
    unittest.main()
