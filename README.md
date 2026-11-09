# Object Detection with YOLOv8

A small personal learning project: running the ready-made **YOLOv8** model on a road video to detect cars, using Python, the Ultralytics library and OpenCV.

It came out of my summer 2025 internship, where I compared Roboflow's ready-made YOLOv8 model with a model I trained myself, then tried to run the ready-made model directly in my own code, without Roboflow. This repository contains that script and a sample video.

> This is a learning exercise, not a production tool. See [What I observed](#what-i-observed) and [Limitations](#limitations) for what it can and cannot do.

## What the script does

1. Loads the small pre-trained model `yolov8n.pt` (it is downloaded automatically if the file is missing).
2. Opens a video file and reads it frame by frame with OpenCV.
3. Sends each frame to YOLOv8, keeping only the **car** class (class 2 in the 80-class COCO set).
4. Ignores detections with confidence of 50% or lower.
5. Draws a box and a label such as `car [74.02%]` on each remaining detection and shows the result in a resizable window.
6. Press **q** to quit.

## Files

| File | Description |
| --- | --- |
| `CODE.py` | The detection script |
| `SAMPLE VDO.mp4` | Sample video for testing the script |
| `yolov8n.pt` | Pre-trained YOLOv8 nano weights from Ultralytics |
| `README.md` | This file |

## How to run

Requirements: Python 3 with the packages below.

```bash
pip install ultralytics opencv-python numpy
python CODE.py
```

**Before running:** the script opens the video path written in step 2 of `CODE.py`. Change it to the sample video in this repository:

```python
cap = cv2.VideoCapture("SAMPLE VDO.mp4")
```

You can also point it to your own video. To detect other COCO classes, change `classes=[2]` (for example `5` is bus and `7` is truck) or remove the argument to detect all 80 classes.

## What I observed

These notes are from my own test on one road video, on my own computer. They are observations, not a measured evaluation.

- **It works without Roboflow.** Once the model is downloaded, no account or website is needed.
- **It was slow.** Detection took about 85 ms per frame in the log, and playing an 11-second clip took roughly 45 seconds, about four times slower than real time. The source video is 4K and the script processes and displays every frame.
- **Confidence was moderate.** In the frames I checked, the confidence values were roughly 56–77%, lower than the ready-made model in Roboflow (mostly 75–95% for larger cars).
- **Some cars were missed.** Because only the car class is kept and everything at or below 50% is dropped, some cars (for example a close white SUV in one frame) did not get a box.

## Limitations

- Only the car class is detected, so trucks, buses and motorbikes are ignored.
- The 50% confidence cut-off is fixed and was not tuned.
- It uses the smallest model (`yolov8n`) and I have not tested larger ones.
- It runs frame by frame at the original video size and does not save an output video.
- I have not tried a GPU.

## Detection screenshot from test on sample video
<img width="1917" height="1017" alt="image" src="https://github.com/user-attachments/assets/a2bbe32d-671e-45d4-9897-828a7494aaa4" />


## Ideas to try next

- Lower the confidence cut-off (the Ultralytics default is 0.25) and compare the results.
- Try a larger model such as `yolov8s.pt` or `yolov8m.pt`.
- Resize frames before detection, or skip frames, to speed things up.
- Run on a GPU.
- Save the annotated video with `cv2.VideoWriter` so it plays smoothly.
- Fine-tune a model on labelled images for objects that the ready-made model does not know.

## Credits and licence note

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which provides the model and library. Ultralytics releases it under its own licence (AGPL-3.0 at the time of writing). Please check their site for the current terms before reusing the weights or code.
- [OpenCV](https://opencv.org/) for video reading and drawing.
- [Roboflow](https://roboflow.com/), used for the comparison with a ready-made model.

## สรุปภาษาไทย

โปรเจกต์เรียนรู้ส่วนตัวที่ใช้โมเดล YOLOv8 สำเร็จรูป (`yolov8n.pt`) ตรวจจับ **รถยนต์** ในวิดีโอถนน ด้วย Python ไลบรารี Ultralytics และ OpenCV โดยไม่ต้องผ่าน Roboflow ใช้โค้ดวนอ่านวิดีโอทีละเฟรม เก็บเฉพาะคลาสรถที่ความมั่นใจมากกว่า 50% แล้ววาดกรอบลงบนภาพ

ก่อนรันให้แก้พาธวิดีโอใน `CODE.py` เป็น `SAMPLE VDO.mp4` แล้วรัน `python CODE.py` กด `q` เพื่อออก

จากการทดลองของผมบนวิดีโอคลิปเดียว: ใช้งานได้ แต่ช้ากว่าเวลาจริงราวสี่เท่าในคอมของผม ความมั่นใจอยู่ที่ราว 56–77% ในเฟรมที่ตรวจดู และมีรถบางคันที่ตรวจไม่เจอ เป็นข้อสังเกตจากการทดลอง ไม่ใช่การวัดผลอย่างเป็นทางการ
