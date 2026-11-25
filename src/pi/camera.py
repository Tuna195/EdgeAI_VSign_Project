"""Đọc ảnh từ camera Pi (qua rpicam-vid) hoặc từ file video, trả về frame BGR.

Vì sao dùng rpicam-vid qua ống (pipe):
  picamera2 cài bằng apt chỉ chạy với Python 3.13 của hệ thống, còn mediapipe 0.10.18
  cần venv Python 3.12. Nên ta gọi chương trình rpicam-vid, bảo nó ghi ảnh thô YUV420
  ra stdout, rồi Python đọc từng khối byte (mỗi khối = 1 frame).

Dùng:
    cam = open_source("cam", 640, 480, 30)        # camera Pi
    cam = open_source("data/videos/a.mp4")        # file video
    ok, frame = cam.read()                        # frame: numpy uint8 (H, W, 3), BGR
    cam.close()
"""
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


class RpicamSource:
    """Camera Pi qua rpicam-vid. Ảnh KHÔNG lật gương (không dùng --hflip)."""

    def __init__(self, width=640, height=480, fps=30, extra_args=None):
        if shutil.which("rpicam-vid") is None:
            raise RuntimeError("Không tìm thấy rpicam-vid (chỉ có trên Raspberry Pi OS).")
        self.width, self.height, self.fps = width, height, fps
        # YUV420 (I420): Y đủ kích thước + U, V mỗi kênh 1/4 -> 1.5 byte/điểm ảnh
        self.frame_bytes = width * height * 3 // 2
        cmd = [
            "rpicam-vid", "-n", "-t", "0",           # không xem trước, chạy mãi
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps),
            "--codec", "yuv420",                     # ảnh thô, không nén
            "-o", "-",                               # ghi ra stdout
        ] + list(extra_args or [])
        # thông báo của rpicam-vid ghi vào file tạm, để in ra khi camera lỗi
        self.log = tempfile.NamedTemporaryFile("w+", prefix="rpicam_", suffix=".log", delete=False)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=self.log, bufsize=0)

    def _read_exact(self, n):
        # Đọc đúng n byte (ống có thể trả về ít hơn mỗi lần đọc)
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.proc.stdout.readinto(view[got:])
            if not k:
                return None
            got += k
        return buf

    def read(self):
        buf = self._read_exact(self.frame_bytes)
        if buf is None:
            self.print_log()
            return False, None
        yuv = np.frombuffer(buf, np.uint8).reshape(self.height * 3 // 2, self.width)
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def print_log(self, n=15):
        try:
            with open(self.log.name) as f:
                lines = f.read().strip().splitlines()
            if lines:
                print("--- rpicam-vid báo:\n" + "\n".join(lines[-n:]))
        except OSError:
            pass

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class VideoFileSource:
    """File video qua OpenCV. Dùng để đo lặp lại trên cùng một đầu vào."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được video: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        return self.cap.read()

    def close(self):
        self.cap.release()


def open_source(source, width=640, height=480, fps=30, extra_args=None):
    """'cam' -> camera Pi; đường dẫn khác -> file video."""
    if source == "cam":
        return RpicamSource(width, height, fps, extra_args)
    return VideoFileSource(source)
