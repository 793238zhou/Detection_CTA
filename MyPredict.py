"""
TASK (optional) is one of [detect, segment, classify, pose]. If it is not passed explicitly YOLOv8 will try to guess the TASK from the model type.
MODE (required) is one of [train, val, predict, export, track, benchmark]
ARGS (optional) are any number of custom arg=value pairs like imgsz=320 that override defaults. For a full list of available ARGS see the Configuration page and defaults.yaml GitHub source.


The Results object contains the following components:

Results.boxes: Boxes object with properties and methods for manipulating bounding boxes
Results.masks: Masks object for indexing masks or getting segment coordinates
Results.probs: torch.Tensor containing class probabilities or logits
Results.orig_img: Original image loaded in memory
Results.path: Path containing the path to the input image

results = model("./ultralytics/assets/bus.jpg")
for result in results:
    # Detection
    result.boxes.xyxy   # box with xyxy format, (N, 4)
    result.boxes.xywh   # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf   # confidence score, (N, 1)
    result.boxes.cls    # cls, (N, 1)

    # Segmentation
    result.masks.data      # masks, (N, H, W)
    result.masks.xy        # x,y segments (pixels), List[segment] * N
    result.masks.xyn       # x,y segments (normalized), List[segment] * N

    # Classification
    result.probs     # cls prob, (num_class, )
"""


### 目标检测：图片
# from ultralytics import YOLO
# import cv2
#
# model = YOLO("yolov8n.pt", task='detect')
# # b = model("./ultralytics/assets/bus.jpg")
# # a = model.predict("./ultralytics/assets/bus.jpg", save=True, imgsz=320, conf=0.5) ?? 与直接使用Model的区别在哪里？
#
# results = model("./ultralytics/assets/bus.jpg")
# # print(results[0].boxes.xyxy)
# # print(results[0].boxes.cls)
# res = results[0].plot()
# # Display the annotated frame
# cv2.imshow("YOLOv8 Inference", res)
# cv2.waitKey(0)


### 目标检测：视频
# import cv2
# from ultralytics import YOLO
#
# # Load the YOLOv8 model
# model = YOLO('yolov8n.pt',task='detect')
# # Open the video file
# video_path = "1.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#
#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)
#
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
#
#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break
#
# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()


# ### 目标分割：图片
# from ultralytics import YOLO
# import cv2
# # Load a model
# model = YOLO('yolov8n-seg.pt')  # load an official model
#
# # Predict with the model
# results = model('./ultralytics/assets/bus.jpg')  # predict on an image
# res = results[0].plot(boxes=False)
# # Display the annotated frame
# cv2.imshow("YOLOv8 Inference", res)
# cv2.waitKey(0)

### 目标分割：视频
# import cv2
# from ultralytics import YOLO
#
# # Load the YOLOv8 model
# model = YOLO('yolov8n-seg.pt', task='segment')
# # Open the video file
# video_path = "1.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#
#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)
#
#         # Visualize the results on the frame
#         # annotated_frame = results[0].plot()
#         annotated_frame = results[0].plot(boxes=False)
#
#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break
#
# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()



### 目标追踪：视频
# from ultralytics import YOLO
#
# # Load a model
# model = YOLO('yolov8n.pt',task='detect')  # load an official detection model
# # model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
#
# # Track with the model
# results = model.track(source="1.mp4", show=True)
# # bytetrack.yaml相当于是tracker的配置文件, 在 ultralytics/tracker/cfg的目录下，可自行在源文件修改，或者创建新的yaml文件
# results = model.track(source="1.mp4", show=True, tracker="bytetrack.yaml")



###  benchmark使用：用于查看并判断哪种格式的模型性能指标更好
# from ultralytics.yolo.utils.benchmarks import benchmark
#
# # Benchmark:用于度量不同Format格式的模型性能指标
# benchmark(model='yolov8n.pt', imgsz=640, half=False, device='cpu')

"""
这些信息可以帮助用户根据其对速度和准确性的要求，为其特定用例选择最佳导出格式。
Benchmarks complete for yolov8n.pt on coco8.yaml at imgsz=640 (47.38s)
                   Format Status  Size (MB)  metrics/mAP50-95(B)  Inference time (ms/im)
0                 PyTorch               6.2               0.6381                  102.97
1             TorchScript              12.4               0.6092                  250.21
2                    ONNX               0.0                  NaN                     NaN
3                OpenVINO               0.0                  NaN                     NaN
4                TensorRT               0.0                  NaN                     NaN
5                  CoreML               0.0                  NaN                     NaN
6   TensorFlow SavedModel               0.0                  NaN                     NaN
7     TensorFlow GraphDef               0.0                  NaN                     NaN
8         TensorFlow Lite               0.0                  NaN                     NaN
9     TensorFlow Edge TPU               0.0                  NaN                     NaN
10          TensorFlow.js               0.0                  NaN                     NaN
11           PaddlePaddle               0.0                  NaN                     NaN
"""

### 分类任务：图片

# from ultralytics import YOLO
# import cv2
# import numpy as np
#
# # Load a model
# # model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-cls.pt', task='classify')  # load a pretrained model (recommended for training)
# # model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights
#
# # Train the model
# # model.train(data='mnist160', epochs=100, imgsz=64)
#
# # Predict with the model
# results = model('./ultralytics/assets/bus.jpg')  # predict on an image

# res = results[0].plot()
# # Display the annotated frame
# cv2.imshow("YOLOv8 Inference", res)
# cv2.waitKey(0)

"""
得到最大概率类别坐标的方法
# res = results[0].probs.numpy()
# index = np.argmax(res)
# print(index)
# print(res[index])
"""



### 姿态检测：图片
# from ultralytics import YOLO
# import cv2
# # Load a model
# # model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
# # model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights
#
# # Train the model
# # model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
# results = model('./ultralytics/assets/bus.jpg')
# res = results[0].plot()
# # Display the annotated frame
# cv2.imshow("YOLOv8 Inference", res)
# cv2.waitKey(0)


### 姿态检测：视频
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt', task='pose')
print('111')
# Open the video file
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()