from ultralytics import YOLO

# Load a model
model = YOLO("yolo12n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model
# model = YOLO("yolo11n-seg.pt")

results = model.train(data="coco128.yaml", epochs=5)


# Validate the model
metrics = model.val()
print("Mean Average Precision for boxes:", metrics.box.map)
print("Mean Average Precision for masks:", metrics.seg.map)

# # Predict with the model
# results = model(["/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/cell_00225.png", "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/cell_00849.png"])  # predict on an image
#
# # Access the results
# i = 0
# for result in results:
# #    xy = result.masks.xy  # mask in polygon format
# #    xyn = result.masks.xyn  # normalized
# #    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename=f"data/result_{i}.jpg")  # save to disk
#     i += 1

