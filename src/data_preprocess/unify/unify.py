from src.datasets.datasets import *

###
# Clean up formatting and outliers, etc.
# Keep all the files in the same place, same name and file type
#
# TODO: Unify the file structure, so each DataSet class logic (categorize, blacklist, etc.) is not necessary.
###

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

def load_sanitized(dataset, filepath, category):
    image = dataset.load(filepath)
    rescale = category not in [MASK, SYNTHETIC]
    sanitized = sanitize(image, rescale)
    if (image.dtype != sanitized.dtype):
        #print("WARNING: " + str(image.dtype) + " -> " + str(sanitized.dtype) + ("(rescale)" if rescale else "") + ", " + filepath)
        pass
    return sanitized

def unify_file(dataset, category):
    # The dataset knows how to standardize its own image files
    def unify(source, target):
        mask = load_sanitized(dataset, source, category)
        save_image(target, mask)
    return unify

def main():
    dataset = DataSet("./data")
    for sample in tqdm(dataset, desc="Unifying datasets"):
        safely_process([], unify_file(dataset, IMAGE))(sample.raw_image, sample.image)
        safely_process([], unify_file(dataset, MASK))(sample.raw_mask, sample.mask)
    for sample in tqdm(list(dataset.unlabeled()), desc="Unifying unlabeled dataset"):
        safely_process([], unify_file(dataset, IMAGE))(sample.raw_image, sample.image)
    
if __name__ == '__main__':
    main()