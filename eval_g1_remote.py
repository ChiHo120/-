"""
eval_g1_remote.py

服务器端远程 PI0 推理版本 —— 图像流改为 TCP 远程接收。

与官方 eval_g1.py 的唯一区别：
    - 图像不再从服务器本机 /dev/video* 读取
    - 改为从机器人 robot_image_stream_server.py (port 5560) 通过 TCP 接收
    - 其余逻辑 (BridgeClient / PI0 推理 / ActionChunk / 限幅) 完全保留

推荐先 dry-run：
    ROBOT_BRIDGE_IP=100.100.204.250 \
    ROBOT_BRIDGE_PORT=5555 \
    ROBOT_IMAGE_PORT=5560 \
    REMOTE_DRY_RUN=1 \
    REMOTE_ENABLE_GRIPPER=0 \
    REMOTE_USE_REALTIME_IMAGE=1 \
    python unitree_lerobot/eval_robot/eval_g1_remote.py \
        --policy.path=.../checkpoints/100000/pretrained_model \
        --repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
        --root="" \
        --episodes=0 \
        --frequency=30 \
        --arm="G1_29" \
        --ee="dex3" \
        --visualization=true
"""

import os
import json
import time
import socket
import logging
import threading

import torch
import logging_mp
import numpy as np

from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)

from unitree_lerobot.eval_robot.utils.utils import (
    predict_action,
    EvalRealConfig,
)

from unitree_lerobot.eval_robot.utils.rerun_visualizer import (
    RerunLogger,
    visualization_data,
)

# 新增：从 TCP 图像流接图
from image_stream_client import ImageStreamClient


# =============================================================================
# Logging
# =============================================================================

try:
    logging_mp.basicConfig(level=logging_mp.INFO)
except RuntimeError as e:
    if "already been started" not in str(e):
        raise

logger_mp = logging_mp.getLogger(__name__)


# =============================================================================
# 环境变量配置
# =============================================================================

ROBOT_BRIDGE_IP = os.environ.get("ROBOT_BRIDGE_IP", "100.100.204.250")
ROBOT_BRIDGE_PORT = int(os.environ.get("ROBOT_BRIDGE_PORT", "5555"))
ROBOT_IMAGE_PORT = int(os.environ.get("ROBOT_IMAGE_PORT", "5560"))

REMOTE_DRY_RUN = os.environ.get("REMOTE_DRY_RUN", "1") == "1"
REMOTE_ENABLE_GRIPPER = os.environ.get("REMOTE_ENABLE_GRIPPER", "0") == "1"
REMOTE_USE_REALTIME_IMAGE = os.environ.get("REMOTE_USE_REALTIME_IMAGE", "0") == "1"

# 图像帧超过此秒数视为过期
REMOTE_IMAGE_MAX_AGE = float(os.environ.get("REMOTE_IMAGE_MAX_AGE", "1.0"))

G1_NUM_MOTOR = 29
ACTION_DIM = 16

GRIPPER_MIN = 0.0
GRIPPER_MAX = 5.0
GRIPPER_CHANGE_THRESHOLD = float(os.environ.get("REMOTE_GRIPPER_THRESHOLD", "0.15"))
GRIPPER_MIN_SEND_INTERVAL = float(os.environ.get("REMOTE_GRIPPER_INTERVAL", "0.50"))

MAX_ACTION_DELTA = float(os.environ.get("REMOTE_MAX_ACTION_DELTA", "0.005"))
REMOTE_DISABLE_ACTION_LIMIT = os.environ.get("REMOTE_DISABLE_ACTION_LIMIT", "0") == "1"

REMOTE_CHUNK_EXEC_STEPS = int(os.environ.get("REMOTE_CHUNK_EXEC_STEPS", "3"))
REMOTE_ACTION_SMOOTH_ALPHA = float(os.environ.get("REMOTE_ACTION_SMOOTH_ALPHA", "0.2"))

# --- 起始姿态 reset (关键) ---
# 是否开机先把手臂移动到训练数据集的起始姿态
REMOTE_RESET_TO_INIT_POSE = os.environ.get("REMOTE_RESET_TO_INIT_POSE", "1") == "1"
# reset 动作每步最大变化 (比推理时的 DELTA 大，但依然很保守)
REMOTE_RESET_STEP_DELTA = float(os.environ.get("REMOTE_RESET_STEP_DELTA", "0.01"))
# reset 到位阈值 (所有关节都在此阈值内视为到位)
REMOTE_RESET_TOL = float(os.environ.get("REMOTE_RESET_TOL", "0.03"))
# reset 超时秒数，防止死循环
REMOTE_RESET_TIMEOUT = float(os.environ.get("REMOTE_RESET_TIMEOUT", "15.0"))
# reset 控制频率
REMOTE_RESET_HZ = float(os.environ.get("REMOTE_RESET_HZ", "30.0"))


# =============================================================================
# Bridge TCP Client  (和官方版本相同)
# =============================================================================

class BridgeClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(3.0)
        self.sock.connect((host, port))
        self.sock.settimeout(0.5)

        self._recv_buf = b""
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()

        self._latest_state = None
        self._latest_state_time = 0.0

        self._stop = False
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="bridge_recv_loop"
        )
        self._thread.start()

    def _recv_loop(self):
        while not self._stop:
            try:
                chunk = self.sock.recv(65536)
                if not chunk:
                    logger_mp.warning("[BridgeClient] socket closed by bridge.")
                    self._stop = True
                    break

                self._recv_buf += chunk
                while b"\n" in self._recv_buf:
                    line, self._recv_buf = self._recv_buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    if msg.get("type") == "state":
                        with self._lock:
                            self._latest_state = msg
                            self._latest_state_time = time.time()

            except socket.timeout:
                continue
            except OSError as e:
                if not self._stop:
                    logger_mp.warning(f"[BridgeClient] recv error: {e}")
                self._stop = True
                break
            except Exception as e:
                if not self._stop:
                    logger_mp.warning(f"[BridgeClient] unexpected recv error: {e}")
                self._stop = True
                break

    def wait_first_state(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._latest_state is not None:
                    return self._latest_state
            time.sleep(0.01)
        raise TimeoutError("Timeout waiting for first bridge state.")

    def get_latest_state(self, max_age=2.0):
        with self._lock:
            state = self._latest_state
            state_time = self._latest_state_time

        if state is None:
            return None, None

        age = time.time() - state_time
        if age > max_age:
            logger_mp.warning(f"[BridgeClient] latest state too old: {age:.3f}s")
            return None, age
        return state, age

    def send_json(self, msg):
        payload = (json.dumps(msg) + "\n").encode("utf-8")
        with self._send_lock:
            self.sock.sendall(payload)

    def close(self):
        self._stop = True
        try:
            self.sock.close()
        except Exception:
            pass


# =============================================================================
# TCP 发送工具 (和官方版本相同)
# =============================================================================

def send_arm_q29(client: BridgeClient, q29):
    if len(q29) != G1_NUM_MOTOR:
        raise ValueError(f"q29 length must be 29, got {len(q29)}")
    q_arr = np.asarray(q29, dtype=np.float32)
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q29 contains NaN or Inf.")
    client.send_json({
        "type": "cmd",
        "q": [float(x) for x in q_arr.tolist()],
        "tau": [0.0] * G1_NUM_MOTOR,
        "mode_pr": 0,
        "client_send_wall_time": time.time(),
    })


def send_gripper(client: BridgeClient, left_angle, right_angle):
    client.send_json({
        "type": "gripper",
        "left_angle": float(np.clip(left_angle, GRIPPER_MIN, GRIPPER_MAX)),
        "right_angle": float(np.clip(right_angle, GRIPPER_MIN, GRIPPER_MAX)),
        "client_send_wall_time": time.time(),
    })


class GripperPublisher:
    def __init__(
        self,
        change_threshold=GRIPPER_CHANGE_THRESHOLD,
        min_send_interval=GRIPPER_MIN_SEND_INTERVAL,
    ):
        self.change_threshold = change_threshold
        self.min_send_interval = min_send_interval
        self.last_left = None
        self.last_right = None
        self.last_send_time = 0.0

    def maybe_send(self, client: BridgeClient, left_angle, right_angle, force=False):
        left_angle = float(np.clip(left_angle, GRIPPER_MIN, GRIPPER_MAX))
        right_angle = float(np.clip(right_angle, GRIPPER_MIN, GRIPPER_MAX))
        now = time.time()

        if not force and (now - self.last_send_time) < self.min_send_interval:
            return False

        need_send = force or self.last_left is None
        if self.last_left is not None:
            if abs(left_angle - self.last_left) > self.change_threshold:
                need_send = True
            if abs(right_angle - self.last_right) > self.change_threshold:
                need_send = True

        if not need_send:
            return False

        send_gripper(client, left_angle, right_angle)
        self.last_left = left_angle
        self.last_right = right_angle
        self.last_send_time = now
        return True


# =============================================================================
# Action Chunk Buffer (和官方版本相同)
# =============================================================================

class ActionChunkBuffer:
    def __init__(self, keep_first_n=3):
        self.keep_first_n = keep_first_n
        self.actions = []
        self.ptr = 0
        self.chunk_id = 0

    def empty(self):
        return self.ptr >= len(self.actions)

    def remaining(self):
        return max(0, len(self.actions) - self.ptr)

    def set_chunk(self, action_chunk):
        arr = np.asarray(action_chunk, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"action_chunk must be 2D, got {arr.shape}")
        if arr.shape[1] < ACTION_DIM:
            raise ValueError(f"action dim must >= 16, got {arr.shape}")

        arr = arr[:, :ACTION_DIM]
        if not np.all(np.isfinite(arr)):
            raise ValueError("action_chunk contains NaN or Inf.")

        n = min(self.keep_first_n, arr.shape[0])
        self.actions = [arr[i].copy() for i in range(n)]
        self.ptr = 0
        self.chunk_id += 1

    def pop(self):
        if self.empty():
            return None
        action = self.actions[self.ptr]
        self.ptr += 1
        return action


# =============================================================================
# 状态 / 动作映射 (和官方版本相同)
# =============================================================================

def bridge_state_to_state16(state_msg):
    q29 = state_msg["q"]
    if len(q29) != G1_NUM_MOTOR:
        raise ValueError(f"bridge q length must be 29, got {len(q29)}")

    gripper = state_msg.get("gripper", {"left": 0.0, "right": 0.0})
    state16 = np.zeros(ACTION_DIM, dtype=np.float32)
    state16[0:7] = np.asarray(q29[15:22], dtype=np.float32)
    state16[7:14] = np.asarray(q29[22:29], dtype=np.float32)
    state16[14] = float(gripper.get("left", 0.0))
    state16[15] = float(gripper.get("right", 0.0))
    return state16


def action16_to_q29(action16, current_q29):
    action16 = np.asarray(action16, dtype=np.float32).reshape(-1)
    if action16.shape[0] < ACTION_DIM:
        raise ValueError(f"action length must >= 16, got {action16.shape}")
    action16 = action16[:ACTION_DIM]
    if not np.all(np.isfinite(action16)):
        raise ValueError("action16 contains NaN or Inf.")
    if len(current_q29) != G1_NUM_MOTOR:
        raise ValueError(f"current_q29 length must be 29, got {len(current_q29)}")

    q29 = [float(x) for x in current_q29]
    q29[15:22] = [float(x) for x in action16[0:7]]
    q29[22:29] = [float(x) for x in action16[7:14]]
    return q29


def safe_action_from_state_and_target(state16, target_action16):
    state16 = np.asarray(state16, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    target_action16 = np.asarray(target_action16, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    if not np.all(np.isfinite(target_action16)):
        raise ValueError("target_action16 contains NaN or Inf.")

    safe = state16.copy()
    if REMOTE_DISABLE_ACTION_LIMIT:
        safe[0:14] = target_action16[0:14]
    else:
        safe[0:14] = state16[0:14] + np.clip(
            target_action16[0:14] - state16[0:14],
            -MAX_ACTION_DELTA, MAX_ACTION_DELTA,
        )

    safe[14] = float(np.clip(target_action16[14], GRIPPER_MIN, GRIPPER_MAX))
    safe[15] = float(np.clip(target_action16[15], GRIPPER_MIN, GRIPPER_MAX))
    return safe.astype(np.float32)


# =============================================================================
# 图像处理
# =============================================================================

def normalize_image_to_hwc(img):
    """
    输出 torch.Tensor, shape=[H, W, 3], float32, range 0~1。
    LeRobot 的 preprocessor 后面会自己处理 resize / CHW。
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Unsupported image type: {type(img)}")

    img = img.detach().clone()

    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 3:
        raise ValueError(f"Image must be 3D, got {tuple(img.shape)}")

    if img.shape[-1] == 3:
        img = img.contiguous()
    elif img.shape[0] == 3:
        img = img.permute(1, 2, 0).contiguous()
    else:
        raise ValueError(f"Cannot infer image layout, got {tuple(img.shape)}")

    img = img.float()
    if float(img.max()) > 2.0:
        img = img / 255.0
    img = torch.clamp(img, 0.0, 1.0)
    return img


def build_image_observation_from_dataset(fallback_sample):
    image_keys = [
        "observation.images.cam_left_high",
        "observation.images.cam_right_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]

    observation = {}
    for key in image_keys:
        if key not in fallback_sample:
            raise KeyError(f"Missing image key in fallback_sample: {key}")
        observation[key] = normalize_image_to_hwc(fallback_sample[key])
    return observation


# =============================================================================
# 起始姿态 reset —— 让机器人走到训练数据集的起始位置
# =============================================================================

def extract_init_state16_from_dataset(dataset):
    """
    从数据集第一个样本里提取 observation.state 作为初始姿态 (16 维)。
    state16: [左臂7, 右臂7, 左夹爪, 右夹爪]
    """
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    sample = dataset[from_idx]

    if "observation.state" not in sample:
        raise KeyError("dataset sample missing observation.state")

    s = sample["observation.state"]
    if isinstance(s, torch.Tensor):
        s = s.detach().cpu().numpy()
    s = np.asarray(s, dtype=np.float32).reshape(-1)

    if s.shape[0] < ACTION_DIM:
        raise ValueError(f"dataset state dim {s.shape} < {ACTION_DIM}")

    return s[:ACTION_DIM].copy()


def reset_to_init_pose(
    bridge_client: BridgeClient,
    target_state16: np.ndarray,
    gripper_pub: GripperPublisher,
    step_delta: float = REMOTE_RESET_STEP_DELTA,
    tol: float = REMOTE_RESET_TOL,
    timeout: float = REMOTE_RESET_TIMEOUT,
    hz: float = REMOTE_RESET_HZ,
    dry_run: bool = False,
    enable_gripper: bool = False,
):
    """
    慢慢把机器人从当前姿态移动到训练起始姿态。

    算法：每步朝目标方向移动 step_delta，直到 14 个手臂关节
    都落在 tol 范围内，或者超时。

    夹爪到位后发一次。
    """
    logger_mp.info(
        f"[Reset] Target init pose (first 16): "
        f"L={np.round(target_state16[0:7], 3).tolist()} "
        f"R={np.round(target_state16[7:14], 3).tolist()} "
        f"G=({target_state16[14]:.3f}, {target_state16[15]:.3f})"
    )
    logger_mp.info(
        f"[Reset] step_delta={step_delta}, tol={tol}, "
        f"timeout={timeout}s, hz={hz}, dry_run={dry_run}"
    )

    dt = 1.0 / hz
    t_start = time.time()
    step_count = 0

    while True:
        if time.time() - t_start > timeout:
            logger_mp.warning(
                f"[Reset] TIMEOUT after {timeout}s. "
                "Continuing anyway. 检查 step_delta 是否过小或机器人是否卡住。"
            )
            break

        state_msg, _ = bridge_client.get_latest_state(max_age=2.0)
        if state_msg is None:
            time.sleep(0.05)
            continue

        current_q29 = state_msg["q"]
        current_state16 = bridge_state_to_state16(state_msg)

        # 差值
        diff = target_state16[0:14] - current_state16[0:14]
        max_abs_diff = float(np.max(np.abs(diff)))

        # 到位判断
        if max_abs_diff < tol:
            logger_mp.info(
                f"[Reset] Arrived after {step_count} steps, "
                f"max_diff={max_abs_diff:.4f} rad."
            )
            break

        # 朝目标走一步
        step = np.clip(diff, -step_delta, step_delta)
        next_arm = current_state16[0:14] + step

        target16 = current_state16.copy()
        target16[0:14] = next_arm
        target16[14] = float(target_state16[14])
        target16[15] = float(target_state16[15])

        q29 = action16_to_q29(target16, current_q29)

        if not dry_run:
            send_arm_q29(bridge_client, q29)

        # 每 30 步打印一次
        if step_count % 30 == 0:
            logger_mp.info(
                f"[Reset step {step_count}] "
                f"max_diff={max_abs_diff:.4f} "
                f"L_cur={np.round(current_state16[0:7], 3).tolist()} "
                f"L_tgt={np.round(target_state16[0:7], 3).tolist()}"
            )

        step_count += 1
        time.sleep(dt)

    # 到位后处理夹爪
    if not dry_run and enable_gripper:
        logger_mp.info(
            f"[Reset] Sending gripper init: "
            f"left={target_state16[14]:.3f}, right={target_state16[15]:.3f}"
        )
        gripper_pub.maybe_send(
            bridge_client,
            target_state16[14], target_state16[15],
            force=True,
        )

    logger_mp.info("[Reset] Done. Pausing 1s before PI0 inference ...")
    time.sleep(1.0)


# =============================================================================
# 【关键】远程图像提供器 —— 替换官方的 RealTimeImageProvider
# =============================================================================

class RemoteImageProvider:
    """
    从机器人 TCP 图像流 (port 5560) 接收 4 路图像。

    行为：
        - 启动时连接 ImageStreamClient，后台线程持续收图
        - get_observation() 非阻塞取最新一帧
        - 图像过期 / 还没收到时，fallback 到 dataset 样本
    """

    SHORT_NAME_MAP = {
        "observation.images.cam_left_high":  "cam_left_high",
        "observation.images.cam_right_high": "cam_right_high",
        "observation.images.cam_left_wrist": "cam_left_wrist",
        "observation.images.cam_right_wrist": "cam_right_wrist",
    }

    def __init__(self, image_host, image_port, max_age=REMOTE_IMAGE_MAX_AGE):
        self.use_realtime = REMOTE_USE_REALTIME_IMAGE
        self.max_age = max_age
        self._client = None

        if not self.use_realtime:
            logger_mp.info(
                "[Image] REMOTE_USE_REALTIME_IMAGE=0, using dataset fallback."
            )
            return

        try:
            self._client = ImageStreamClient(image_host, image_port)
            self._client.start()
            logger_mp.info(
                f"[Image] ImageStreamClient connecting to {image_host}:{image_port} ..."
            )
        except Exception as e:
            logger_mp.warning(
                f"[Image] ImageStreamClient init failed: {e}. Fallback to dataset."
            )
            self.use_realtime = False
            self._client = None

    def wait_first_frame(self, timeout=10.0):
        """等图像流来第一帧，非必须。"""
        if not self.use_realtime or self._client is None:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._client.get_latest(max_age=5.0) is not None:
                stats = self._client.stats()
                logger_mp.info(f"[Image] First remote frame ready. stats={stats}")
                return True
            time.sleep(0.1)
        logger_mp.warning("[Image] Timeout waiting for first remote frame.")
        return False

    def get_observation(self, fallback_sample):
        """
        返回 (observation_dict, realtime_ok, image_cost_sec)
        observation_dict 的每张图是 torch.Tensor [H, W, 3] float32, 0~1, RGB
        """
        if not self.use_realtime or self._client is None:
            return build_image_observation_from_dataset(fallback_sample), False, 0.0

        t0 = time.perf_counter()
        images_bgr = self._client.get_latest(max_age=self.max_age)

        if images_bgr is None:
            stats = self._client.stats()
            logger_mp.warning(
                f"[Image] No fresh remote image (frames={stats['frame_count']}, "
                f"age={stats['age']}). Fallback to dataset."
            )
            return build_image_observation_from_dataset(fallback_sample), False, 0.0

        observation = {}
        for full_key, short_name in self.SHORT_NAME_MAP.items():
            if short_name not in images_bgr:
                logger_mp.warning(
                    f"[Image] missing {short_name} in remote packet, fallback."
                )
                return build_image_observation_from_dataset(fallback_sample), False, 0.0

            bgr = images_bgr[short_name]
            # BGR -> RGB
            rgb = bgr[..., ::-1].copy()
            observation[full_key] = normalize_image_to_hwc(rgb)

        image_cost = time.perf_counter() - t0
        return observation, True, image_cost

    def close(self):
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass


# =============================================================================
# Remote eval 主循环 (和官方版本几乎相同，只是 image_provider 换了)
# =============================================================================

def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")
    logger_mp.info(f"ROBOT_BRIDGE_IP={ROBOT_BRIDGE_IP}")
    logger_mp.info(f"ROBOT_BRIDGE_PORT={ROBOT_BRIDGE_PORT}")
    logger_mp.info(f"ROBOT_IMAGE_PORT={ROBOT_IMAGE_PORT}")
    logger_mp.info(f"REMOTE_DRY_RUN={REMOTE_DRY_RUN}")
    logger_mp.info(f"REMOTE_ENABLE_GRIPPER={REMOTE_ENABLE_GRIPPER}")
    logger_mp.info(f"REMOTE_USE_REALTIME_IMAGE={REMOTE_USE_REALTIME_IMAGE}")
    logger_mp.info(f"REMOTE_IMAGE_MAX_AGE={REMOTE_IMAGE_MAX_AGE}")
    logger_mp.info(f"REMOTE_DISABLE_ACTION_LIMIT={REMOTE_DISABLE_ACTION_LIMIT}")
    logger_mp.info(f"MAX_ACTION_DELTA={MAX_ACTION_DELTA}")
    logger_mp.info(f"REMOTE_CHUNK_EXEC_STEPS={REMOTE_CHUNK_EXEC_STEPS}")
    logger_mp.info(f"REMOTE_ACTION_SMOOTH_ALPHA={REMOTE_ACTION_SMOOTH_ALPHA}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    bridge_client = None
    image_provider = None

    try:
        # 连接 bridge
        bridge_client = BridgeClient(ROBOT_BRIDGE_IP, ROBOT_BRIDGE_PORT)
        logger_mp.info(
            f"Connected to bridge: {ROBOT_BRIDGE_IP}:{ROBOT_BRIDGE_PORT}"
        )

        first_state = bridge_client.wait_first_state(timeout=5.0)
        logger_mp.info(
            "First bridge state: "
            f"L={np.round(first_state['q'][15:22], 3).tolist()} "
            f"R={np.round(first_state['q'][22:29], 3).tolist()}"
        )

        gripper_pub = GripperPublisher()
        action_buffer = ActionChunkBuffer(keep_first_n=REMOTE_CHUNK_EXEC_STEPS)

        # dataset fallback
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        fallback_sample = dataset[from_idx]
        task = fallback_sample["task"]
        logger_mp.info(f"Using dataset frame index as fallback: {from_idx}")
        logger_mp.info(f"Task: {task}")

        # 远程图像提供器
        image_provider = RemoteImageProvider(
            image_host=ROBOT_BRIDGE_IP,
            image_port=ROBOT_IMAGE_PORT,
            max_age=REMOTE_IMAGE_MAX_AGE,
        )
        image_provider.wait_first_frame(timeout=10.0)

        # =================================================================
        # Reset 到训练起始姿态
        # =================================================================
        if REMOTE_RESET_TO_INIT_POSE and not REMOTE_DRY_RUN:
            try:
                init_state16 = extract_init_state16_from_dataset(dataset)
                logger_mp.info(
                    f"[Reset] Dataset init pose extracted: "
                    f"L={np.round(init_state16[0:7], 3).tolist()} "
                    f"R={np.round(init_state16[7:14], 3).tolist()} "
                    f"G=({init_state16[14]:.3f}, {init_state16[15]:.3f})"
                )

                user_input = input(
                    "Enter 'r' to reset to init pose, 's' to skip reset and start: "
                )

                if user_input.lower() == "r":
                    reset_to_init_pose(
                        bridge_client=bridge_client,
                        target_state16=init_state16,
                        gripper_pub=gripper_pub,
                        step_delta=REMOTE_RESET_STEP_DELTA,
                        tol=REMOTE_RESET_TOL,
                        timeout=REMOTE_RESET_TIMEOUT,
                        hz=REMOTE_RESET_HZ,
                        dry_run=False,
                        enable_gripper=REMOTE_ENABLE_GRIPPER,
                    )
                elif user_input.lower() == "s":
                    logger_mp.info("[Reset] Skipped by user.")
                else:
                    logger_mp.info("Unknown input. Exit.")
                    return

            except Exception as e:
                logger_mp.warning(f"[Reset] Failed: {e}. Continuing without reset.")

        # 用户确认开始推理
        user_input = input("Enter 's' to start PI0 inference: ")
        if user_input.lower() != "s":
            logger_mp.info("User did not enter 's'. Exit.")
            return

        logger_mp.info(f"Starting remote evaluation at {cfg.frequency} Hz.")

        idx = 0
        last_action16 = None
        last_timing_print = 0.0

        latency_stats = {
            "image": [], "infer": [], "send": [], "loop": [], "state_age": [],
        }

        while True:
            loop_start_time = time.perf_counter()

            # 1) 最新状态
            state_msg, state_age = bridge_client.get_latest_state(max_age=2.0)
            if state_msg is None:
                logger_mp.warning("[Remote] No fresh state, skip.")
                time.sleep(0.05)
                continue

            current_q29 = state_msg["q"]
            state16 = bridge_state_to_state16(state_msg)
            state_tensor = torch.from_numpy(state16).float()

            # 2) 远程图像
            observation, realtime_ok, image_cost = image_provider.get_observation(
                fallback_sample
            )
            observation["observation.state"] = state_tensor

            if idx == 0:
                for k, v in observation.items():
                    if "images" in k:
                        logger_mp.info(
                            f"[Image Check] {k}: shape={tuple(v.shape)}, "
                            f"dtype={v.dtype}, min={float(v.min()):.3f}, "
                            f"max={float(v.max()):.3f}"
                        )
                logger_mp.info(
                    f"[State Check] observation.state: "
                    f"shape={tuple(state_tensor.shape)}, dtype={state_tensor.dtype}"
                )

            # 3) PI0 推理 / chunk
            infer_cost = 0.0
            if action_buffer.empty():
                t_infer0 = time.perf_counter()
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    task,
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                infer_cost = time.perf_counter() - t_infer0
                action_np = action.detach().cpu().numpy()
                action_buffer.set_chunk(action_np)

            raw_action16 = action_buffer.pop()
            if raw_action16 is None:
                logger_mp.warning("[Remote] action buffer empty, skip.")
                continue

            # 4) 低通滤波
            if last_action16 is not None:
                raw_action16 = (
                    REMOTE_ACTION_SMOOTH_ALPHA * raw_action16
                    + (1.0 - REMOTE_ACTION_SMOOTH_ALPHA) * last_action16
                )
            last_action16 = raw_action16.copy()

            # 5) 限幅
            action16 = safe_action_from_state_and_target(
                state16=state16, target_action16=raw_action16,
            )
            q29 = action16_to_q29(action16, current_q29)

            # 6) 发送
            send_cost = 0.0
            if not REMOTE_DRY_RUN:
                t_send0 = time.perf_counter()
                send_arm_q29(bridge_client, q29)
                if REMOTE_ENABLE_GRIPPER:
                    sent = gripper_pub.maybe_send(
                        bridge_client, action16[14], action16[15],
                        force=(idx == 0),
                    )
                    if sent:
                        logger_mp.info(
                            f"Gripper sent: left={action16[14]:.3f}, "
                            f"right={action16[15]:.3f}"
                        )
                send_cost = time.perf_counter() - t_send0

            # 7) 可视化
            if cfg.visualization:
                visualization_data(
                    idx, observation, state_tensor.numpy(), action16, rerun_logger,
                )

            # 延迟统计
            loop_cost = time.perf_counter() - loop_start_time
            latency_stats["image"].append(image_cost)
            latency_stats["infer"].append(infer_cost)
            latency_stats["send"].append(send_cost)
            latency_stats["loop"].append(loop_cost)
            latency_stats["state_age"].append(0.0 if state_age is None else state_age)

            for k in latency_stats:
                if len(latency_stats[k]) > 100:
                    latency_stats[k] = latency_stats[k][-100:]

            if idx % max(1, int(cfg.frequency)) == 0:
                logger_mp.info(
                    f"[Remote {idx}] "
                    f"L={np.round(action16[0:7], 3).tolist()} "
                    f"R={np.round(action16[7:14], 3).tolist()} "
                    f"G=({action16[14]:.3f}, {action16[15]:.3f}) "
                    f"dry_run={REMOTE_DRY_RUN} "
                    f"chunk_remain={action_buffer.remaining()} "
                    f"realtime_image={realtime_ok}"
                )

            now = time.time()
            if now - last_timing_print > 2.0:
                last_timing_print = now
                logger_mp.info(
                    "[Timing] "
                    f"loop={loop_cost:.3f}s "
                    f"real_hz={1.0 / max(loop_cost, 1e-6):.2f} "
                    f"infer={infer_cost:.3f}s "
                    f"image={image_cost:.3f}s "
                    f"send={send_cost:.4f}s "
                    f"state_age={0.0 if state_age is None else state_age:.3f}s "
                    f"avg_loop={float(np.mean(latency_stats['loop'])):.3f}s "
                    f"avg_infer={float(np.mean(latency_stats['infer'])):.3f}s "
                    f"avg_image={float(np.mean(latency_stats['image'])):.3f}s "
                    f"chunk_remain={action_buffer.remaining()}"
                )

            idx += 1

            # 控制频率
            sleep_time = (1.0 / cfg.frequency) - (
                time.perf_counter() - loop_start_time
            )
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger_mp.info("Remote eval stopped by user.")
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
        raise
    finally:
        if image_provider is not None:
            image_provider.close()
        if bridge_client is not None:
            bridge_client.close()


# =============================================================================
# Main
# =============================================================================

@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    if getattr(cfg, "root", ""):
        dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root)
    else:
        dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of remote eval")


if __name__ == "__main__":
    print("[DEBUG] eval_g1_remote.py main entered")
    init_logging()
    eval_main()
