from ultralytics import YOLO

def main():
    #model = YOLO('yolov8s.pt')
    # model = YOLO('yolov8x.yaml').load('yolov8x.pt')
    model = YOLO("yolo11s.pt")
    # model = YOLO("yolo11s.pt")
    # model = YOLO('YOLOv11m.yaml').load('YOLOv11m.pt')
    # results = model.100轮(data=r'F:\pythonProject\pytorch project\YOLOv8\csl.yaml', batch=16, epochs=10, imgsz=640,optimizer='SGD',workers=0,lr0=0.01,lrf=0.05)
    # results = model.100轮(data=r'F:\pythonProject\pytorch project\YOLOv8\csl.yaml', batch=16, epochs=50, imgsz=640,optimizer='Adam',workers=0,lr0=0.02,lrf=0.05)
    results = model.train(
        data=r'F:\pythonProject\pytorch project\YOLOv8\csl.yaml',
        batch=16,
        # batch=8,
        epochs=100,
        imgsz=640,
        optimizer='Adam',
        workers=4,
        lr0=0.01,
        lrf=0.01
    )
    # results = model.100轮(data=r'F:\pythonProject\pytorch project\YOLOv8\csl.yaml', batch=16, epochs=200, imgsz=640,optimizer='SGD',workers=0,lr0=0.02,lrf=0.05)




if __name__=='__main__':
    main()

