from src.datasets.datasets import *

###
# Clean up formatting and outliers, etc.
# Keep all the files in the same place, same name and file type
#
# TODO: Unify the file structure, so each DataSet class logic (categorize, blacklist, etc.) is not necessary.
###

def unify_dataset(dataroot, dataset):
    files = list(dataset.enumerate(dataroot + "/raw"))
    for filepath, cat in tqdm(files, desc="Unifying dataset " + dataset.root):
        folder, name, ext = split_filepath(filepath)
        source = dataroot + "/raw" + filepath
        target_type = ext
        #target_type = ".tiff" if cat == MASK else ".png"
        target = dataroot + "/unify" + folder + name + target_type
        safely_process([], unify_file(dataset))(source, target)

def unify_file(dataset):
    # The dataset knows how to standardize its own image files
    def unify(source, target):
        mask = unify_load(dataset, source)
        save_image(target, mask)
    return unify

def upper_bound(tensor):
    max = tensor.max()
    for bytes in [0, 1, 2, 4, 8]:
        bits = bytes << 3
        if max < (1 << bits):
            return bits

def sanitize(tensor, rescale):
    assert 0 <= tensor.min(), "Tensor contains negative values"
    # Remove alpha channel if present
    if tensor.shape[-1] == 4:  # RGBA
        tensor = color.rgba2rgb(tensor)
    # Scale back to [0, 255] if needed
    bits = upper_bound(tensor)
    if rescale and (bits != 8):
        scaled = tensor * ((1 << 8)/(1 << bits))
        return scaled.astype('uint8')
    return tensor.astype('uint' + str(bits))

def unify_load(dataset, filepath):
    image = dataset.load(filepath)
    category = dataset.categorize(filepath)
    rescale = category not in [MASK, SYNTHETIC]
    sanitized = sanitize(image, rescale)
    if (image.dtype != sanitized.dtype):
        #print("WARNING: " + str(image.dtype) + " -> " + str(sanitized.dtype) + ("(rescale)" if rescale else "") + ", " + filepath)
        pass
    return sanitized

def main():
    for dataset in DataSet.filesets:
        unify_dataset("./data", dataset)
    
if __name__ == '__main__':
    main()