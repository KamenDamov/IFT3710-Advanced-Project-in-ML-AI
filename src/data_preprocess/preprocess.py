from . import normalization
from . import transformations

if __name__ == "__main__":
    normalization.main("./data")
    transformations.main("./data")
