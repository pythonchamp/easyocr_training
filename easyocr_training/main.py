import easyocr


class Reader:
    def __init__(self, language="en",model_dir="./models"):
        self.language = language
        self.model_dir = model_dir
        self.reader = easyocr.Reader([self.language],model_storage_directory=model_dir,download_enabled=False)  # this needs to run only once to load the model into memory

    def readtext(self, image_path):
        result = self.reader.readtext(image_path, detail=0)
        return result
