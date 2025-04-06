import os
import cv2
import pandas as pd
import numpy as np
import random
from imgaug import augmenters as iaa

def generate_synthetic_ocr_data(image_folder, label_csv, output_folder, num_aug_per_image=5):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)

    df = pd.read_csv(label_csv)
    new_data = []

    # Define augmentations
    augmenter = iaa.Sequential([
        iaa.Affine(rotate=(-5, 5), scale=(0.9, 1.1), shear=(-5, 5)),
        iaa.AdditiveGaussianNoise(scale=(5, 20)),
        iaa.MotionBlur(k=3),
        iaa.Multiply((0.8, 1.2)),
        iaa.ContrastNormalization((0.75, 1.25))
    ])

    for idx, row in df.iterrows():
        image_path = os.path.join(image_folder, row['filename'])
        label = row['words']

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Save original image
        base_filename = f"{idx:06}_orig.png"
        cv2.imwrite(os.path.join(output_folder, 'images', base_filename), image)
        new_data.append((base_filename, label))

        # Generate synthetic variants
        for i in range(num_aug_per_image):
            aug_img = augmenter(image=image)
            aug_filename = f"{idx:06}_aug_{i}.png"
            cv2.imwrite(os.path.join(output_folder, 'images', aug_filename), aug_img)
            new_data.append((aug_filename, label))

    # Save updated labels
    df_out = pd.DataFrame(new_data, columns=['filename', 'words'])
    df_out.to_csv(os.path.join(output_folder, 'labels.csv'), index=False)
    print(f"Synthetic data created in: {output_folder}")

if __name__ == '__main__':
    generate_synthetic_ocr_data(
        image_folder='unique_images',
        label_csv='unique_images/labels.csv',
        output_folder='synthetic_data',
        num_aug_per_image=20
    )