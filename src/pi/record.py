"""Quay video test trên Pi, xem trực tiếp trên trình duyệt máy tính, có nhắc kịch bản.

Chạy trên Pi (trong venv):
    python -m src.pi.record --out data/videos/test_30s.mp4
Rồi mở trên máy tính (cùng mạng Wi-Fi/LAN):  http://<IP của Pi>:8000

Luồng hoạt động:
    xem trước --(bấm "Bắt đầu quay")--> đếm ngược 5 s --> quay theo kịch bản --> lưu file, thoát
Kịch bản mặc định (36 s): 3 s tay xuống | 5 s giữ hai tay yên trước ngực | 20 s ký liên tục
                          | 5 s giữ yên | 3 s tay xuống.  Đổi bằng --script "5:...,20:...".
--shutter <micro giây>: đặt thời gian phơi sáng cố định (vd 8000 = 8 ms) để giảm nhoè khi tay
    di chuyển nhanh. Mặc định 0 = camera tự chọn (phòng tối sẽ tự phơi lâu -> dễ nhoè).
    Phơi ngắn làm ảnh tối -> camera tự tăng độ khuếch đại (gain, ảnh nhiễu hơn) -> nên bật đèn.

Đầu ra:
    <out>.mp4  : video gốc, KHÔNG vẽ chữ, KHÔNG lật gương
    <out>.json : thông tin quay + khoảng frame của từng đoạn kịch bản (để benchmark tách
                 thời gian lúc có tay / không có tay) + thời gian phơi sáng / gain thực tế
    <out>_cam.txt : thông số camera từng frame do rpicam-vid ghi (ExposureTime, AnalogueGain...)
Nút "Lật gương" trên trang web chỉ lật hình HIỂN THỊ cho dễ nhìn, không ảnh hưởng file.
"""
import argparse
import json
import os
import re
import socket
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from src.pi.camera import RpicamSource, VideoFileSource, open_source

COUNTDOWN_S = 5

# Trạng thái dùng chung giữa luồng camera và luồng web (bảo vệ bằng khoá)
lock = threading.Lock()
shared = {"jpeg": None, "state": "preview", "status": {}, "start_requested": False}


def parse_script(text):
    """'5:Để tay xuống,20:Ký' -> [(5.0, 'Để tay xuống'), (20.0, 'Ký')]"""
    phases = []
    for part in text.split(","):
        sec, label = part.split(":", 1)
        phases.append((float(sec), label.strip()))
    return phases


def cam_stat(path, key):
    """Trung vị một thông số camera (vd ExposureTime) từ file metadata của rpicam-vid."""
    try:
        with open(path) as f:
            vals = [float(v) for v in re.findall(key + r"\W+([0-9.]+)", f.read())]
    except OSError:
        return None
    if not vals:
        return None
    vals.sort()
    return round(vals[len(vals) // 2], 2)


def local_ip():
    # Lấy IP mạng LAN của Pi (không gửi gói tin nào ra ngoài)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quay video test</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
 #st{font-size:42px;font-weight:bold;padding:12px}
 #sub{font-size:22px;color:#aaa;min-height:30px}
 img{max-width:96vw;max-height:70vh;border:3px solid #444;margin-top:8px}
 img.mirror{transform:scaleX(-1)}
 button{font-size:22px;padding:10px 24px;margin:10px;border-radius:8px;border:0;cursor:pointer}
 #go{background:#2a7;color:#fff} #go:disabled{background:#555}
 .rec{color:#f44}
</style></head><body>
<div id="st">Đang kết nối...</div><div id="sub"></div>
<img id="v" src="/stream"><br>
<button id="go" onclick="fetch('/start',{method:'POST'})">Bắt đầu quay</button>
<button onclick="document.getElementById('v').classList.toggle('mirror')">Lật gương (chỉ màn hình)</button>
<script>
async function poll(){
  try{
    const s = await (await fetch('/status')).json();
    const st = document.getElementById('st'), sub = document.getElementById('sub');
    st.textContent = s.title; sub.textContent = s.detail || '';
    st.className = s.state === 'recording' ? 'rec' : '';
    document.getElementById('go').disabled = s.state !== 'preview';
  }catch(e){ document.getElementById('st').textContent = 'Mất kết nối với Pi'; }
  setTimeout(poll, 250);
}
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    stream_fps = 15  # hình xem trước chỉ cần ~15 fps, đỡ tốn CPU của Pi

    def log_message(self, *args):  # tắt log mỗi request
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/status":
            with lock:
                body = json.dumps(shared["status"], ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
        elif self.path == "/stream":
            # MJPEG: gửi liên tục từng ảnh JPEG, trình duyệt tự hiển thị như video
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpg = shared["jpeg"]
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg + b"\r\n")
                    time.sleep(1.0 / self.stream_fps)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/start":
            with lock:
                if shared["state"] == "preview":
                    shared["start_requested"] = True
            self._send(b"ok", "text/plain")
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description="Quay video test, xem trực tiếp qua trình duyệt")
    ap.add_argument("--out", default="data/videos/test_30s.mp4")
    ap.add_argument("--source", default="cam", help="'cam' hoặc file video (để thử)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--script", default="3:Tay xuống,5:Giữ hai tay yên trước ngực,"
                    "20:Ký liên tục (cả hai tay),5:Giữ hai tay yên trước ngực,3:Tay xuống")
    ap.add_argument("--shutter", type=int, default=0, help="phơi sáng (micro giây), 0 = tự động")
    args = ap.parse_args()

    phases = parse_script(args.script)
    total_s = sum(p[0] for p in phases)
    n_total = int(round(total_s * args.fps))          # dừng theo SỐ FRAME, không theo đồng hồ
    # Khoảng frame của từng đoạn: [start, end)
    bounds, acc = [], 0.0
    for sec, label in phases:
        start = int(round(acc * args.fps))
        acc += sec
        bounds.append({"label": label, "seconds": sec,
                       "start_frame": start, "end_frame": int(round(acc * args.fps))})

    cam_meta = os.path.splitext(args.out)[0] + "_cam.txt"
    extra = ["--metadata", cam_meta, "--metadata-format", "txt"]
    if args.shutter > 0:
        extra += ["--shutter", str(args.shutter)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cam = open_source(args.source, args.width, args.height, args.fps, extra)
    is_file = isinstance(cam, VideoFileSource)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Mở trên máy tính: http://{local_ip()}:{args.port}   (Ctrl+C để thoát)")

    writer, n_written, t_count0, stamps = None, 0, None, []
    last_jpeg_t = 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Hết nguồn ảnh / camera dừng.")
                break
            now = time.perf_counter()
            if is_file:
                time.sleep(1.0 / args.fps)  # chỉ khi thử bằng file: giả lập tốc độ thật

            # Ảnh xem trước: nén JPEG ~15 lần/giây
            if now - last_jpeg_t >= 1.0 / Handler.stream_fps:
                ok_j, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok_j:
                    with lock:
                        shared["jpeg"] = jpg.tobytes()
                last_jpeg_t = now

            with lock:
                state = shared["state"]
                if state == "preview" and shared["start_requested"]:
                    shared["state"] = state = "countdown"
                    t_count0 = now

            if state == "preview":
                status = {"state": state, "title": "Xem trước",
                          "detail": "Chỉnh vị trí: cách camera 1-1.5 m, thấy từ đầu đến thắt lưng. "
                                    "Sẵn sàng thì bấm Bắt đầu quay."}
            elif state == "countdown":
                left = COUNTDOWN_S - (now - t_count0)
                if left <= 0:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.out, fourcc, args.fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Không tạo được file {args.out}")
                    with lock:
                        shared["state"] = state = "recording"
                    print("Bắt đầu quay...")
                else:
                    status = {"state": state, "title": f"Chuẩn bị... {int(left) + 1}",
                              "detail": f"Đoạn đầu: {phases[0][1]}"}

            if state == "recording":
                writer.write(frame)              # ghi ảnh gốc, không vẽ gì lên
                stamps.append(now)
                n_written += 1
                i = next(k for k, b in enumerate(bounds) if n_written <= b["end_frame"])
                b = bounds[i]
                left = (b["end_frame"] - n_written) / args.fps
                nxt = bounds[i + 1]["label"] if i + 1 < len(bounds) else "Hết"
                status = {"state": state, "title": f"● {b['label']}  ({left:.0f} s)",
                          "detail": f"Tiếp theo: {nxt}   |   Tổng: {n_written / args.fps:.0f}/{total_s:.0f} s"}
                if n_written >= n_total:
                    break

            with lock:
                shared["status"] = status
    except KeyboardInterrupt:
        print("\nDừng bởi người dùng.")
    finally:
        cam.close()
        if writer is not None:
            writer.release()

    if n_written:
        real_fps = (n_written - 1) / (stamps[-1] - stamps[0]) if n_written > 1 else 0.0
        meta = {
            "file": os.path.basename(args.out),
            "created": datetime.now().isoformat(timespec="seconds"),
            "source": args.source, "width": args.width, "height": args.height,
            "fps_requested": args.fps, "fps_measured": round(real_fps, 2),
            "frames": n_written, "complete": n_written >= n_total,
            "mirrored": False, "codec": "mp4v",
            "shutter_us": args.shutter or "auto",
            "exposure_us_p50": cam_stat(cam_meta, "ExposureTime"),
            "analogue_gain_p50": cam_stat(cam_meta, "AnalogueGain"),
            "phases": bounds,
        }
        meta_path = os.path.splitext(args.out)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with lock:
            shared["status"] = {"state": "done", "title": "Đã quay xong ✔",
                                "detail": f"{n_written} frame, fps đo được {real_fps:.1f}"}
        print(f"Đã lưu {args.out} ({n_written} frame, fps đo được {real_fps:.1f})")
        print(f"Thông tin đoạn kịch bản: {meta_path}")
        print(f"Phơi sáng thực tế p50: {meta['exposure_us_p50']} µs | gain p50: {meta['analogue_gain_p50']}")
        time.sleep(1.5)  # để trang web kịp hiện "Đã quay xong"
    server.shutdown()


if __name__ == "__main__":
    main()
