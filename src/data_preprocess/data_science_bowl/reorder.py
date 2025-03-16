import os
import shutil

def create_structure(data_split):
    os.makedirs(os.path.join('data', 'cellpose', data_split, 'images'), exist_ok=True)
    os.makedirs(os.path.join('data', 'cellpose', data_split, 'labels'), exist_ok=True)

def move_files(data_split):
    base_dir = os.path.join('data', 'cellpose', data_split)
    for file_name in os.listdir(base_dir):
        old_path = os.path.join(base_dir, file_name)

        if os.path.isdir(old_path):
            continue

        if "imgs" not in file_name:
            new_path = os.path.join(base_dir, 'images', file_name)
        else:
            file_name = file_name.replace("imgs", "img")
            new_path = os.path.join(base_dir, 'labels', file_name)

        shutil.move(old_path, new_path)

def main():
    create_structure("train")
    create_structure("test")
    move_files("train")
    move_files("test")

if __name__ == '__main__':
    main()
