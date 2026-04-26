using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Windows;

namespace CameraProject
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture? _capture;
        private BackgroundSubtractorMOG2? _bgSubtractor;
        private CancellationTokenSource? _cancellationTokenSource;

        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += MainWindow_Loaded;
            this.Closed += MainWindow_Closed;
        }

        private async void MainWindow_Loaded(object? sender, RoutedEventArgs e)
        {
            // Tối ưu hóa FFMPEG để giảm độ trễ (real-time)
            Environment.SetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay");

            // URL đã kiểm tra thành công
            string rtspUrl = "rtsp://admin:%40Lamtran213@192.168.1.184:554/cam/realmonitor?channel=1&subtype=0";
            
            _capture = new VideoCapture(rtspUrl, VideoCaptureAPIs.FFMPEG);
            // Ép buffer size về nhỏ nhất có thể
            _capture.Set(VideoCaptureProperties.BufferSize, 1); 
            
            // Khởi tạo thuật toán MOG2 để nhận diện vùng chuyển động
            _bgSubtractor = BackgroundSubtractorMOG2.Create(history: 500, varThreshold: 16, detectShadows: false);
            
            _cancellationTokenSource = new CancellationTokenSource();
            
            // Chạy vòng lặp lấy frame ở luồng nền (background thread)
            await Task.Run(() => ProcessVideo(_cancellationTokenSource.Token));
        }

        private void ProcessVideo(CancellationToken token)
        {
            if (_capture == null || _bgSubtractor == null) return;

            Mat frame = new Mat();
            Mat fgMask = new Mat();
            
            while (!token.IsCancellationRequested && _capture.IsOpened())
            {
                if (!_capture.Read(frame) || frame.Empty())
                {
                    Thread.Sleep(10);
                    continue;
                }

                // 1. Lấy mask chuyển động
                _bgSubtractor.Apply(frame, fgMask);

                // 2. Lọc nhiễu
                Cv2.Erode(fgMask, fgMask, new Mat(), null, 1);
                Cv2.Dilate(fgMask, fgMask, new Mat(), null, 2);

                // 3. Tìm các contours
                Cv2.FindContours(fgMask, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                foreach (var contour in contours)
                {
                    var area = Cv2.ContourArea(contour);
                    // Chỉ lấy các vùng có diện tích đủ lớn (tránh nhiễu nhỏ, có thể tăng/giảm số này)
                    if (area > 1500)
                    {
                        var rect = Cv2.BoundingRect(contour);
                        // Vẽ khung đỏ bao quanh vùng chuyển động
                        Cv2.Rectangle(frame, rect, Scalar.Red, 2);
                    }
                }

                // 4. Chuyển Mat sang BitmapSource và cập nhật UI
                try
                {
                    var bitmap = frame.ToWriteableBitmap();
                    bitmap.Freeze(); // Cần Freeze() để có thể sử dụng ở UI Thread

                    Dispatcher.Invoke(() =>
                    {
                        CameraImage.Source = bitmap;
                    });
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error rendering frame: " + ex.Message);
                }
                
                // Bỏ Thread.Sleep ở đây để vòng lặp lấy frame chạy nhanh nhất có thể.
                // Hàm _capture.Read(frame) đã tự động block (chờ) cho đến khi có frame mới từ camera.
            }
            
            frame.Dispose();
            fgMask.Dispose();
        }

        private void MainWindow_Closed(object? sender, EventArgs e)
        {
            _cancellationTokenSource?.Cancel();
            _capture?.Release();
            _capture?.Dispose();
            _bgSubtractor?.Dispose();
        }
    }
}