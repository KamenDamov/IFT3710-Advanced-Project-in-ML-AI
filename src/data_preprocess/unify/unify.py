import os
import shutil

def unify_folders(paths, output_folder):
    images_output = os.path.join(output_folder, "images")
    labels_output = os.path.join(output_folder, "labels")
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)

    image_counter = 0

    for path in paths:
        images_path = os.path.join(path, "images")
        labels_path = os.path.join(path, "labels")

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            raise ValueError(f"Missing 'images' or 'labels' folder in {path}")

        for image_file in os.listdir(images_path):
            image_counter += 1
            label_file = os.path.splitext(image_file)[0] + ".png"

            src_image = os.path.join(images_path, image_file)
            src_label = os.path.join(labels_path, label_file)

            if not os.path.exists(src_label):
                raise ValueError(f"Label file {label_file} not found for image {image_file}")

            dst_image = os.path.join(images_output, f"image_{image_counter:05d}.png")
            dst_label = os.path.join(labels_output, f"image_{image_counter:05d}.png")

            shutil.copy(src_image, dst_image)
            shutil.copy(src_label, dst_label)

def main():
    paths = [
        "data\\preprocessing_outputs\\cellpose\\train\\transformed_images_labels",
        "data\\preprocessing_outputs\\data_science_bowl\\transformed_images_labels",
        "data\\preprocessing_outputs\\transformed_images_labels"
    ]
    output_folder = "data\\preprocessing_outputs\\unified_set"
    unify_folders(paths, output_folder)

if __name__ == '__main__':
    main()