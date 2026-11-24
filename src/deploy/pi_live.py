"""Xem MediaPipe chạy trực tiếp trên Pi, từ trình duyệt của laptop (không cần màn hình cắm vào Pi).

Đường đi của một frame:
  camera -> [luồng đọc camera: chỉ giữ frame MỚI NHẤT] -> MediaPipe Hand + Pose -> vẽ bộ xương + số đo
         -> nén JPEG -> [máy chủ web nhỏ] -> trình duyệt trên laptop

Chạy trên Pi (trong venv Python 3.12), từ thư mục gốc repo:
  python -m src.deploy.pi_live                       Pi Camera (CSI) qua rpicam-vid
  python -m src.deploy.pi_live --no-pose             chỉ chạy Hand (để so tốc độ)
  python -m src.deploy.pi_live --source 0            webcam USB / webcam laptop (đọc bằng OpenCV)
  python -m src.deploy.pi_live --source a.mp4        file video, phát lặp lại (thử khi không có camera)
Rồi trên laptop mở trình duyệt: http://<IP của Pi>:8000   (xem IP trên Pi: hostname -I). Dừng: Ctrl+C.
"""
import argparse
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from src.core.landmarks import HAND_DIM, create_hand, create_pose, draw_landmarks, frame_to_vector

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>VSign Pi live</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head><body><img src="/stream"></body></html>"""


class Camera:
    """Đọc camera ở một LUỒNG RIÊNG và chỉ giữ frame mới nhất.

    Vì sao: camera cho 30 frame/giây nhưng MediaPipe trên Pi 4 chỉ xử lý ~4 frame/giây. Nếu đọc lần lượt
    từng frame, frame chưa xử lý dồn lại trong hàng đợi -> hình trễ ngày càng nhiều (vài giây, vài chục giây...).
    Luồng này đọc liên tục và ghi đè frame cũ, nên MediaPipe lúc nào cũng xử lý ảnh vừa chụp.
    """

    def __init__(self, source, width, height, fps):
        self.width, self.height = width, height
        self.lock = threading.Lock()
        self.latest, self.count = None, 0          # frame mới nhất và tổng số frame camera đã đọc
        self.running, self.proc, self.cap = True, None, None
        if source == "rpicam":
            if width % 64:
                raise SystemExit("Chiều rộng nên là bội số của 64 (vd. 640, 320) để ảnh thô YUV không bị lệch dòng.")
            # rpicam-vid ghi ảnh thô YUV420 ra stdout ("-o -"); Python đọc qua ống (pipe) nối với stdout đó
            cmd = ["rpicam-vid", "-t", "0", "-n", "--width", str(width), "--height", str(height),
                   "--framerate", str(fps), "--codec", "yuv420", "-o", "-"]
            self.log = open("rpicam.log", "w")          # thông báo/lỗi của rpicam-vid ghi vào file này
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.log)
        else:
            self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not self.cap.isOpened():
                raise SystemExit(f"Không mở được {source!r}")
            self.is_file = not source.isdigit()
            self.delay = 1 / (self.cap.get(cv2.CAP_PROP_FPS) or 30)   # file: phát đúng tốc độ thật, không đọc vội
        threading.Thread(target=self._loop, daemon=True).start()

    def _read_one(self):
        """Đọc 1 frame -> dữ liệu (bytes YUV hoặc ảnh BGR), hoặc None nếu camera dừng."""
        if self.proc:
            size = self.width * self.height * 3 // 2   # YUV420: Y đủ mọi pixel, U và V mỗi cái 1/4 -> 1.5 byte/pixel
            buf = self.proc.stdout.read(size)
            return buf if len(buf) == size else None
        ok, frame = self.cap.read()
        if not ok and self.is_file:                    # file hết -> quay lại đầu
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if self.is_file:
            time.sleep(self.delay)
        return frame if ok else None

    def _loop(self):
        while self.running:
            data = self._read_one()
            if data is None:
                break
            with self.lock:
                self.latest, self.count = data, self.count + 1
        if self.running:                               # tự dừng, không phải do mình gọi close()
            why = f"rpicam-vid đã thoát, mã {self.proc.wait()} (xem rpicam.log)" if self.proc else "hết frame"
            print(f"[camera] Không đọc được frame nữa: {why}", flush=True)
        self.running = False

    def get(self, last_count):
        """Đợi tới khi có frame mới hơn frame số last_count -> (ảnh BGR, số thứ tự). Camera dừng -> (None, ...)."""
        waited, warned = time.monotonic(), False
        while True:
            with self.lock:
                data, count = self.latest, self.count
            if count > last_count:
                break
            if not self.running:
                return None, last_count
            if not warned and time.monotonic() - waited > 3:
                print("[camera] Đã 3 giây không có frame mới (camera bị treo?)", flush=True)
                warned = True
            time.sleep(0.005)
        if self.proc:                                  # chỉ đổi YUV -> BGR cho frame thật sự dùng, đỡ tốn CPU
            yuv = np.frombuffer(data, np.uint8).reshape(self.height * 3 // 2, self.width)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420), count
        return data.copy(), count                      # copy: luồng đọc có thể ghi đè trong lúc ta vẽ

    def close(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
        if self.cap:
            self.cap.release()


class Stream:
    """Giữ ảnh JPEG mới nhất. Mỗi trình duyệt đang xem chờ ảnh mới rồi gửi đi."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpg, self.id = None, 0

    def publish(self, jpg):
        with self.cond:
            self.jpg, self.id = jpg, self.id + 1
            self.cond.notify_all()                     # đánh thức các luồng đang chờ ảnh mới

    def wait(self, last_id):
        with self.cond:
            self.cond.wait_for(lambda: self.id > last_id, timeout=5)
            return self.jpg, self.id


def make_handler(stream):
    """Máy chủ web: '/' trả trang HTML, '/stream' gửi ảnh JPEG liên tục (MJPEG: trình duyệt tự thay ảnh cũ)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = 0
            try:
                while True:
                    jpg, new_id = stream.wait(last)
                    if jpg is None or new_id == last:  # chưa có ảnh mới (camera chậm/dừng) -> chờ tiếp
                        continue
                    last = new_id
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass                                   # người xem đóng tab

        def log_message(self, *args):                  # không in từng yêu cầu ra terminal
            pass

    return Handler


def put_text(frame, text, y):
    """Chữ trắng viền đen (không dấu: OpenCV không vẽ được chữ có dấu)."""
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="rpicam", help="rpicam (mặc định) | số webcam (0, 1...) | đường dẫn file video")
    p.add_argument("--size", default="640x480", help="kích thước ảnh camera, vd. 640x480 (chỉ dùng với rpicam)")
    p.add_argument("--fps", type=int, default=30, help="FPS của camera (chỉ dùng với rpicam)")
    p.add_argument("--no-pose", action="store_true", help="tắt Pose, chỉ chạy Hand")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    width, height = map(int, args.size.lower().split("x"))

    cam = Camera(args.source, width, height, args.fps)
    hand, pose = create_hand(), None if args.no_pose else create_pose()
    stream = Stream()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(stream))   # 0.0.0.0: máy khác trong mạng vào được
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Đang chạy. Trên laptop mở: http://<IP của Pi>:{args.port}  (xem IP: hostname -I). Ctrl+C để dừng.", flush=True)

    start = time.monotonic()
    count, last_ts, fps, ms_avg = 0, -1, 0.0, 0.0
    prev = last_print = start
    first_count = None
    try:
        while True:
            frame, count = cam.get(count)
            if frame is None:
                print("Camera đã dừng.")
                break
            first_count = first_count or count
            t0 = time.perf_counter()
            last_ts = max(last_ts + 1, int((time.monotonic() - start) * 1000))   # mốc thời gian phải tăng dần
            vec = frame_to_vector(frame, last_ts, hand, pose)
            ms = 1000 * (time.perf_counter() - t0)

            now = time.monotonic()
            fps = 1 / (now - prev) if fps == 0 else 0.9 * fps + 0.1 / max(now - prev, 1e-6)   # làm mượt
            ms_avg = ms if ms_avg == 0 else 0.9 * ms_avg + 0.1 * ms
            prev = now
            cam_fps = (count - first_count) / max(now - start, 1e-6)
            n_hands = int(vec[:HAND_DIM].any()) + int(vec[HAND_DIM:2 * HAND_DIM].any())

            draw_landmarks(frame, vec)
            put_text(frame, f"MediaPipe {ms_avg:.0f} ms | xu ly {fps:.1f} FPS | camera {cam_fps:.0f} FPS", 22)
            put_text(frame, f"{n_hands} tay | {'Hand' if args.no_pose else 'Hand+Pose'}", 46)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            stream.publish(jpg.tobytes())

            if now - last_print >= 2:                  # cứ 2 giây in một dòng ra terminal SSH
                print(f"[{now - start:5.0f}s] MediaPipe {ms_avg:4.0f} ms | xử lý {fps:4.1f} FPS"
                      f" | camera {cam_fps:4.1f} FPS | {n_hands} tay", flush=True)
                last_print = now
    except KeyboardInterrupt:
        pass
    except Exception:                                  # lỗi bất ngờ: in ra stdout để không bị 2>/dev/null giấu mất
        traceback.print_exc(file=sys.stdout)
    finally:
        cam.close()
        server.shutdown()
        hand.close()
        if pose:
            pose.close()


if __name__ == "__main__":
    main()
