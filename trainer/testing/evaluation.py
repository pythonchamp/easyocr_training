import os
import pandas as pd
import easyocr

def evaluate_ocr_to_dataframe(eval_folder, labels_csv_path, model_dir):
    reader = easyocr.Reader(
        ['en'],
        recog_network='best_accuracy',
        model_storage_directory=model_dir,
        user_network_directory=model_dir,
        download_enabled=False,
        verbose=True
    )

    df = pd.read_csv(labels_csv_path)
    results = []

    for idx, row in df.iterrows():
        image_path = os.path.join(eval_folder, row['filename'])
        expected = row['words']

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        prediction = reader.readtext(image_path, detail=0)
        actual = prediction[0] if prediction else ''

        results.append({
            'filename': row['filename'],
            'expected': expected,
            'actual': actual
        })

    result_df = pd.DataFrame(results)
    print(result_df.to_markdown())
    return result_df

if __name__ == '__main__':
    df = evaluate_ocr_to_dataframe(
        eval_folder="unique_images",
        labels_csv_path='unique_images/labels.csv',  # Folder where best_accuracy.pth is stored
        model_dir='../saved_models',  # This is the model name prefix
    )
