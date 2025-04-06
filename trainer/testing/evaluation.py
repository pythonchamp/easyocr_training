import os
import pandas as pd
import easyocr

def evaluate_custom_easyocr_model(
    image_folder,
    model_storage_directory,
    user_network_directory,
    recog_network='custom_model',
    lang_list=['en']
):
    # Initialize EasyOCR with your custom model
    reader = easyocr.Reader(
        lang_list=lang_list,
        model_storage_directory=model_storage_directory,
        user_network_directory=user_network_directory,
        recog_network=recog_network,
        download_enabled=False,
        gpu=False
    )

    results = []
    total = 0
    correct = 0

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

    for filename in sorted(os.listdir(image_folder)):
        if not filename.lower().endswith(supported_exts):
            continue

        expected_text = os.path.splitext(filename)[0]
        image_path = os.path.join(image_folder, filename)
        ocr_result = reader.readtext(image_path, detail=0)

        predicted_text = ''.join(ocr_result).strip()

        match = expected_text == predicted_text
        if match:
            correct += 1
        total += 1

        results.append({
            'image_file': filename,
            'expected': expected_text,
            'predicted': predicted_text,
            'match': match
        })

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n✅ Trained EasyOCR Accuracy: {correct}/{total} ({accuracy:.2f}%)")

    df = pd.DataFrame(results)
    print(df)

    return df


if __name__ == '__main__':
    df = evaluate_custom_easyocr_model(
        image_folder="en_val",
        model_storage_directory="../saved_models/en_filtered",  # Folder where best_accuracy.pth is stored
        user_network_directory="../saved_models/en_filtered",  # Usually the same
        recog_network='custom_model',  # This is the model name prefix
        lang_list=['en']  # Or your custom language code
    )
