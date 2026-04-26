# CameraProject

## Nhan dien khung xuong va tu the voi YOLOv8 (Python + WPF)

Ung dung WPF hien camera RTSP va ve skeleton realtime bang YOLOv8 Pose.
Nhan dien cac nhan co ban:

- `dung`
- `ngoi`
- `ngu`
- `nga`
- `te`
- `vay_tay`

## Cau hinh nhanh

1. Cai Python 3.10+ va dam bao chay duoc lenh `python` trong terminal.
2. Cai thu vien Python:

```powershell
cd .\CameraProject
python -m pip install -r .\python\requirements.txt
```

3. Build va chay app:

```powershell
dotnet build .\CameraProject.slnx
dotnet run --project .\CameraProject\CameraProject.csproj
```

## Luu y

- URL RTSP hien tai duoc khai bao trong [CameraProject/MainWindow.xaml.cs](CameraProject/MainWindow.xaml.cs#L11).
- Python worker nam tai [CameraProject/python/pose_worker.py](CameraProject/python/pose_worker.py).
- Model su dung file `yolov8n-pose.onnx` trong project va duoc copy sang output khi build.
- Bo quy tac nhan dien `ngu / ngoi / te / nga / vay_tay` la heuristic co ban, ban co the tinh chinh trong ham `classify_pose`.