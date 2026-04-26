using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Windows;

namespace CameraProject
{
    public partial class MainWindow : System.Windows.Window
    {
        private const string RtspUrl = "rtsp://admin:%40Lamtran213@192.168.1.184:554/cam/realmonitor?channel=1&subtype=0";

        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        private static readonly (int From, int To)[] SkeletonEdges =
        {
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        };

        private VideoCapture? _capture;
        private CancellationTokenSource? _cancellationTokenSource;

        private Process? _pythonProcess;
        private StreamWriter? _pythonInputWriter;
        private Task? _pythonStdoutTask;
        private Task? _pythonStderrTask;

        private readonly object _poseLock = new();
        private PoseResponse? _latestPose;
        private long _frameId;

        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += MainWindow_Loaded;
            this.Closed += MainWindow_Closed;
        }

        private async void MainWindow_Loaded(object? sender, RoutedEventArgs e)
        {
            UpdateStatus("Dang khoi dong YOLOv8 Pose...");

            Environment.SetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay");

            try
            {
                if (!StartPythonPoseWorker())
                {
                    UpdateStatus("Khong the chay Python worker. Kiem tra Python + ultralytics.", true);
                    return;
                }

                _capture = new VideoCapture(RtspUrl, VideoCaptureAPIs.FFMPEG);
                _capture.Set(VideoCaptureProperties.BufferSize, 1);

                if (!_capture.IsOpened())
                {
                    UpdateStatus("Khong mo duoc RTSP stream.", true);
                    return;
                }

                _cancellationTokenSource = new CancellationTokenSource();
                _ = Task.Run(() => ProcessVideo(_cancellationTokenSource.Token));
                UpdateStatus("Dang nhan dien skeleton + tu the realtime.");
            }
            catch (Exception ex)
            {
                UpdateStatus($"Loi khoi dong: {ex.Message}", true);
            }

            await Task.CompletedTask;
        }

        private bool StartPythonPoseWorker()
        {
            string baseDirectory = AppContext.BaseDirectory;
            string scriptPath = Path.Combine(baseDirectory, "python", "pose_worker.py");
            string modelPath = Path.Combine(baseDirectory, "yolov8n-pose.pt");
            if (!File.Exists(modelPath))
            {
                modelPath = Path.Combine(baseDirectory, "yolov8n-pose.onnx");
            }

            if (!File.Exists(scriptPath))
            {
                UpdateStatus($"Khong tim thay script Python: {scriptPath}", true);
                return false;
            }

            if (!File.Exists(modelPath))
            {
                UpdateStatus("Khong tim thay model .pt/.onnx trong thu muc output.", true);
                return false;
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{scriptPath}\" --model \"{modelPath}\" --conf 0.35",
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            _pythonProcess = new Process { StartInfo = startInfo };

            try
            {
                if (!_pythonProcess.Start())
                {
                    return false;
                }

                _pythonInputWriter = _pythonProcess.StandardInput;
                _pythonStdoutTask = Task.Run(ReadPythonStdoutLoop);
                _pythonStderrTask = Task.Run(ReadPythonStderrLoop);
                return true;
            }
            catch (Exception ex)
            {
                UpdateStatus($"Khong chay duoc python.exe: {ex.Message}", true);
                return false;
            }
        }

        private void ReadPythonStdoutLoop()
        {
            if (_pythonProcess == null)
            {
                return;
            }

            while (!_pythonProcess.HasExited)
            {
                string? line = _pythonProcess.StandardOutput.ReadLine();
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                try
                {
                    PoseResponse? response = JsonSerializer.Deserialize<PoseResponse>(line, JsonOptions);
                    if (response == null)
                    {
                        continue;
                    }

                    lock (_poseLock)
                    {
                        _latestPose = response;
                    }
                }
                catch
                {
                    // Keep loop alive even when one JSON line is malformed.
                }
            }
        }

        private void ReadPythonStderrLoop()
        {
            if (_pythonProcess == null)
            {
                return;
            }

            while (!_pythonProcess.HasExited)
            {
                string? line = _pythonProcess.StandardError.ReadLine();
                if (!string.IsNullOrWhiteSpace(line))
                {
                    Dispatcher.Invoke(() => { PoseHintText.Text = line; });
                }
            }
        }

        private void ProcessVideo(CancellationToken token)
        {
            if (_capture == null)
            {
                return;
            }

            using Mat frame = new();

            while (!token.IsCancellationRequested && _capture.IsOpened())
            {
                if (!_capture.Read(frame) || frame.Empty())
                {
                    Thread.Sleep(8);
                    continue;
                }

                _frameId++;

                // Send frame to Python every 2 frames to reduce CPU overhead.
                if (_frameId % 2 == 0)
                {
                    SendFrameToPython(frame, _frameId);
                }

                PoseResponse? pose;
                lock (_poseLock)
                {
                    pose = _latestPose;
                }

                if (pose != null)
                {
                    DrawPose(frame, pose);
                }

                try
                {
                    var bitmap = frame.ToWriteableBitmap();
                    bitmap.Freeze();

                    Dispatcher.Invoke(() =>
                    {
                        CameraImage.Source = bitmap;
                    });
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error rendering frame: {ex.Message}");
                }
            }
        }

        private void SendFrameToPython(Mat frame, long frameId)
        {
            if (_pythonInputWriter == null || _pythonProcess == null || _pythonProcess.HasExited)
            {
                return;
            }

            try
            {
                Cv2.ImEncode(".jpg", frame, out byte[] jpegBuffer, new[]
                {
                    (int)ImwriteFlags.JpegQuality,
                    70
                });

                string base64Frame = Convert.ToBase64String(jpegBuffer);
                var request = new FrameRequest
                {
                    FrameId = frameId,
                    ImageB64 = base64Frame
                };

                string payload = JsonSerializer.Serialize(request);
                _pythonInputWriter.WriteLine(payload);
                _pythonInputWriter.Flush();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Send frame to Python failed: {ex.Message}");
            }
        }

        private static void DrawPose(Mat frame, PoseResponse response)
        {
            foreach (PersonPose person in response.People)
            {
                if (person.Keypoints.Count < 17)
                {
                    continue;
                }

                foreach ((int from, int to) in SkeletonEdges)
                {
                    float[] p1 = person.Keypoints[from];
                    float[] p2 = person.Keypoints[to];

                    if (p1.Length < 3 || p2.Length < 3 || p1[2] < 0.25f || p2[2] < 0.25f)
                    {
                        continue;
                    }

                    Cv2.Line(
                        frame,
                        new OpenCvSharp.Point((int)p1[0], (int)p1[1]),
                        new OpenCvSharp.Point((int)p2[0], (int)p2[1]),
                        new Scalar(20, 210, 20),
                        2);
                }

                foreach (float[] kp in person.Keypoints)
                {
                    if (kp.Length >= 3 && kp[2] >= 0.25f)
                    {
                        Cv2.Circle(frame, new OpenCvSharp.Point((int)kp[0], (int)kp[1]), 3, new Scalar(0, 220, 255), -1);
                    }
                }

                if (person.Bbox?.Length == 4)
                {
                    int x1 = Math.Max(0, (int)person.Bbox[0]);
                    int y1 = Math.Max(0, (int)person.Bbox[1]);
                    int x2 = Math.Min(frame.Width - 1, (int)person.Bbox[2]);
                    int y2 = Math.Min(frame.Height - 1, (int)person.Bbox[3]);

                    Cv2.Rectangle(frame, new OpenCvSharp.Rect(x1, y1, Math.Max(1, x2 - x1), Math.Max(1, y2 - y1)), new Scalar(0, 140, 255), 2);

                    string label = string.IsNullOrWhiteSpace(person.Label) ? "nguoi" : person.Label;
                    Cv2.PutText(
                        frame,
                        $"{label} {person.Confidence:0.00}",
                        new OpenCvSharp.Point(x1, Math.Max(20, y1 - 8)),
                        HersheyFonts.HersheySimplex,
                        0.65,
                        new Scalar(40, 255, 255),
                        2);
                }
            }
        }

        private void UpdateStatus(string message, bool error = false)
        {
            Dispatcher.Invoke(() =>
            {
                StatusText.Text = message;
                StatusText.Foreground = error
                    ? System.Windows.Media.Brushes.OrangeRed
                    : System.Windows.Media.Brushes.LightGreen;
            });
        }

        private async void MainWindow_Closed(object? sender, EventArgs e)
        {
            _cancellationTokenSource?.Cancel();

            _capture?.Release();
            _capture?.Dispose();

            try
            {
                if (_pythonInputWriter != null)
                {
                    await _pythonInputWriter.FlushAsync();
                    _pythonInputWriter.Close();
                }
            }
            catch
            {
                // Ignore errors while shutting down.
            }

            if (_pythonProcess != null && !_pythonProcess.HasExited)
            {
                _pythonProcess.Kill(entireProcessTree: true);
                _pythonProcess.Dispose();
            }

            if (_pythonStdoutTask != null)
            {
                await _pythonStdoutTask;
            }

            if (_pythonStderrTask != null)
            {
                await _pythonStderrTask;
            }
        }
    }

    public sealed class FrameRequest
    {
        public long FrameId { get; set; }
        public string ImageB64 { get; set; } = string.Empty;
    }

    public sealed class PoseResponse
    {
        public long FrameId { get; set; }
        public List<PersonPose> People { get; set; } = new();
    }

    public sealed class PersonPose
    {
        public float[]? Bbox { get; set; }
        public float Confidence { get; set; }
        public string Label { get; set; } = string.Empty;
        public List<float[]> Keypoints { get; set; } = new();
    }
}