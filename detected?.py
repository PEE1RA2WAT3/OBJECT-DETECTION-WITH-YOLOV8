import numpy as np
import cv2
from ultralytics import YOLO  # นำเข้าไลบรารี YOLOv8

# 1. โหลดโมเดล YOLOv8
# โมเดลจะดาวน์โหลดไฟล์ 'yolov8n.pt' มาให้เองอัตโนมัติในครั้งแรก
model = YOLO("yolov8n.pt")

# ดึงรายชื่อคลาสทั้งหมดจากโมเดล (มี 80 คลาส รวมถึง บุคคล, รถ, สัตว์ ฯลฯ)
CLASSES = model.names

# สุ่มสีสำหรับแต่ละคลาสเพื่อใช้สร้างกรอบ
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 2. เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture("GX010042.MP4.mp4")

# สร้างหน้าต่างที่สามารถใช้เมาส์ปรับขนาดได้
cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL) 
# สามารถกำหนดขนาดเริ่มต้นของหน้าต่างได้ด้วย (เช่น กว้าง 800 x สูง 600)
cv2.resizeWindow("YOLOv8 Detection", 1600, 1200) 

while True:
    ret, frame = cap.read()
    if not ret:
        break  # ออกจากลูปเมื่อวิดีโอจบ

    # 3. ส่งภาพเข้าไปตรวจจับใน YOLOv8
    # stream=True ช่วยให้ประมวลผลวิดีโอได้ลื่นไหลและประหยัดแรม
    results = model(frame, classes=[2], stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # ดึงค่าความมั่นใจ (Confidence)
            percent = float(box.conf[0])

            # กรองเอาเฉพาะวัตถุที่มีความมั่นใจมากกว่า 50% (0.5)
            if percent > 0.5:
                # ดึงพิกัดของกรอบ [startX, startY, endX, endY]
                x1, y1, x2, y2 = box.xyxy[0]
                startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)

                # ดึงดัชนีของคลาสวัตถุที่ตรวจจับได้
                class_index = int(box.cls[0])

                # สร้างข้อความแสดงผล เช่น "person [92.50%]"
                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent * 100)

                # วาดกรอบสี่เหลี่ยมและใส่พื้นหลังข้อความเหมือนโค้ดเดิมของคุณ
                color = COLORS[class_index]
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.rectangle(
                    frame,
                    (startX - 1, startY - 30),
                    (endX + 1, startY),
                    color,
                    cv2.FILLED,
                )

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    frame,
                    label,
                    (startX + 20, y + 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
    
    # 4. แสดงผลลัพธ์
    cv2.imshow("YOLOv8 Detection", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
