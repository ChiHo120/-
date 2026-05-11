# image_stream_client.py
# 运行位置：推理服务器
#
# 功能：
#   - 后台线程持续从机器人 robot_image_stream_server.py (port 5560) 接收 4 路图像
#   - 主线程通过 get_latest() 非阻塞拿最新一帧，不阻塞推理主循环
#   - 自动重连：网络/摄像头抖动不会让整个推理挂掉

import socket
import struct
import json
import threading
import time
import numpy as np
import cv2


def recv_exact(sock, n):
    """保证读满 n 字节，处理 TCP 分片"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while recv")
        buf += chunk
    return buf


def parse_packet(packet_bytes):
    """
    解包 robot_image_stream_server 的格式:
      [4B header_len][header_json][body: concat jpegs]
    返回:
      header dict, {cam_name: ndarray(HWC BGR)}
    """
    header_len = struct.unpack("!I", packet_bytes[:4])[0]
    header = json.loads(packet_bytes[4:4 + header_len].decode("utf-8"))
    body = packet_bytes[4 + header_len:]

    images = {}
    for key, info in header["images"].items():
        off = info["offset"]
        size = info["size"]
        jpg = body[off:off + size]
        arr = np.frombuffer(jpg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise RuntimeError(f"imdecode failed for {key}")
        images[key] = img

    return header, images


class ImageStreamClient:
    """
    后台线程持续从机器人接收图像流。
    主线程通过 get_latest() 非阻塞获取最新一帧。
    """

    def __init__(self, host, port=5560, reconnect_interval=2.0):
        self.host = host
        self.port = port
        self.reconnect_interval = reconnect_interval

        self._lock = threading.Lock()
        self._latest_images = None       # dict[str, ndarray(BGR)]
        self._latest_t = None            # 机器人侧时间戳
        self._latest_recv_wall = None    # 服务器侧接收时间
        self._frame_count = 0

        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="image_stream_client"
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop_event.is_set():
            sock = None
            try:
                print(f"[ImageClient] Connecting to {self.host}:{self.port} ...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self.host, self.port))
                sock.settimeout(3.0)
                print(f"[ImageClient] Connected to {self.host}:{self.port}")

                while not self._stop_event.is_set():
                    # 读 4 字节包长
                    len_bytes = recv_exact(sock, 4)
                    packet_len = struct.unpack("!I", len_bytes)[0]

                    if packet_len <= 0 or packet_len > 50 * 1024 * 1024:
                        raise RuntimeError(f"invalid packet_len={packet_len}")

                    packet = recv_exact(sock, packet_len)
                    header, images = parse_packet(packet)

                    with self._lock:
                        self._latest_images = images
                        self._latest_t = header.get("t", None)
                        self._latest_recv_wall = time.time()
                        self._frame_count += 1

            except Exception as e:
                print(f"[ImageClient] Error: {e}, reconnect in "
                      f"{self.reconnect_interval}s")
                time.sleep(self.reconnect_interval)

            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

    def get_latest(self, max_age=1.0):
        """
        返回最新一帧: dict[str, ndarray(HWC BGR)]
        max_age: 超过该秒数视为过期，返回 None
        """
        with self._lock:
            if self._latest_images is None:
                return None
            if self._latest_recv_wall is None:
                return None
            age = time.time() - self._latest_recv_wall
            if age > max_age:
                return None
            return dict(self._latest_images)

    def stats(self):
        with self._lock:
            age = None
            if self._latest_recv_wall is not None:
                age = time.time() - self._latest_recv_wall
            return {
                "frame_count": self._frame_count,
                "latest_t": self._latest_t,
                "latest_recv_wall": self._latest_recv_wall,
                "age": age,
                "has_latest": self._latest_images is not None,
            }


# --------- 独立调试入口：只接图像并打印，不碰控制 ---------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--show", action="store_true", help="cv2.imshow 四路拼图")
    args = parser.parse_args()

    client = ImageStreamClient(args.ip, args.port)
    client.start()

    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            imgs = client.get_latest()
            stats = client.stats()

            if imgs is None:
                print(f"[debug] no image yet, frames={stats['frame_count']}")
            else:
                shapes = {k: v.shape for k, v in imgs.items()}
                print(f"[debug] frames={stats['frame_count']} "
                      f"age={stats['age']:.3f}s shapes={shapes}")

                if args.show:
                    keys = ["cam_left_high", "cam_right_high",
                            "cam_left_wrist", "cam_right_wrist"]
                    tiles = []
                    for k in keys:
                        if k in imgs:
                            tiles.append(imgs[k])
                    if len(tiles) == 4:
                        top = np.hstack([tiles[0], tiles[1]])
                        bot = np.hstack([tiles[2], tiles[3]])
                        grid = np.vstack([top, bot])
                        cv2.imshow("4-cam", grid)
                        if cv2.waitKey(1) == ord('q'):
                            break

            time.sleep(0.2)
    finally:
        client.stop()
        if args.show:
            cv2.destroyAllWindows()
