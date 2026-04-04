"""
DashScope Wan 2.7 Video Editing API client.
Async task submission + polling for video-to-video ad replacement.
"""

import os
import time
import logging
import threading
from typing import Optional, Callable
from dataclasses import dataclass, field

import requests

logger = logging.getLogger("adstream.wan_client")

DASHSCOPE_ENDPOINTS = {
    "singapore": "https://dashscope-intl.aliyuncs.com/api/v1",
    "beijing": "https://dashscope.aliyuncs.com/api/v1",
    "virginia": "https://dashscope-us.aliyuncs.com/api/v1",
}

DEFAULT_PROMPT_TEMPLATE = (
    "The {product} turns into {branded} branded {item} and appears naturally "
    "in the scene. Keep the original visual style. My movement, audio and "
    "behavior don't change."
)


@dataclass
class WanTask:
    task_id: str
    status: str = "PENDING"
    video_url: Optional[str] = None
    error: Optional[str] = None
    submitted_at: float = 0.0
    completed_at: float = 0.0
    metadata: dict = field(default_factory=dict)


class WanClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        region: str = "singapore",
        model: str = "wan2.7-videoedit",
        resolution: str = "720P",
        poll_interval: float = 10.0,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY not set")

        base = DASHSCOPE_ENDPOINTS.get(region, DASHSCOPE_ENDPOINTS["singapore"])
        self.submit_url = f"{base}/services/aigc/video-generation/video-synthesis"
        self.task_url_template = f"{base}/tasks/{{}}"
        self.model = model
        self.resolution = resolution
        self.poll_interval = poll_interval

        self._pending: dict[str, WanTask] = {}
        self._completed: dict[str, WanTask] = {}
        self._poll_thread: Optional[threading.Thread] = None
        self._polling = False
        self._callbacks: dict[str, Callable] = {}

    def submit(
        self,
        video_url: str,
        prompt: str,
        duration: int = 0,
        reference_images: Optional[list[str]] = None,
        on_complete: Optional[Callable] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[WanTask]:
        if not self.api_key:
            logger.error("No API key configured")
            return None

        media = [{"type": "video", "url": video_url}]
        if reference_images:
            for img_url in reference_images[:3]:
                media.append({"type": "reference_image", "url": img_url})

        payload = {
            "model": self.model,
            "input": {
                "prompt": prompt,
                "media": media,
            },
            "parameters": {
                "resolution": self.resolution,
                "prompt_extend": False,
                "watermark": False,
                "audio_setting": "origin",
            },
        }
        if duration > 0:
            payload["parameters"]["duration"] = max(2, min(10, duration))

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable",
        }

        try:
            resp = requests.post(
                self.submit_url, json=payload, headers=headers, timeout=30
            )
            data = resp.json()
            print(f"[WAN SUBMIT] status={resp.status_code} response={data}")
            resp.raise_for_status()

            task_id = data["output"]["task_id"]
            task = WanTask(
                task_id=task_id,
                status=data["output"]["task_status"],
                submitted_at=time.time(),
                metadata=metadata or {},
            )
            self._pending[task_id] = task

            if on_complete:
                self._callbacks[task_id] = on_complete

            logger.info("Wan task submitted: %s", task_id)
            return task

        except requests.RequestException as e:
            logger.error("Wan submit failed: %s", e)
            return None
        except (KeyError, ValueError) as e:
            logger.error("Wan response parse error: %s", e)
            return None

    def poll_task(self, task_id: str) -> WanTask:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = self.task_url_template.format(task_id)

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            data = resp.json()
            print(f"[WAN POLL] task={task_id} response={data}")
            resp.raise_for_status()

            output = data.get("output", {})
            status = output.get("task_status", "UNKNOWN")

            task = self._pending.get(task_id) or WanTask(task_id=task_id)
            task.status = status

            if status == "SUCCEEDED":
                task.video_url = output.get("video_url")
                task.completed_at = time.time()
                elapsed = task.completed_at - task.submitted_at
                print(
                    f"[WAN DONE] task={task_id} elapsed={elapsed:.1f}s url={task.video_url}"
                )

            elif status == "FAILED":
                task.error = output.get("message", "Unknown error")
                task.completed_at = time.time()
                print(f"[WAN FAILED] task={task_id} error={task.error}")

            return task

        except requests.RequestException as e:
            logger.error("Wan poll error: %s", e)
            task = self._pending.get(task_id) or WanTask(task_id=task_id)
            task.status = "POLL_ERROR"
            return task

    def start_polling(self):
        if self._polling:
            return
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self):
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None

    def get_completed(self) -> dict[str, WanTask]:
        result = dict(self._completed)
        self._completed.clear()
        return result

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def download_video(self, url: str, output_path: str) -> bool:
        print(f"[WAN DOWNLOAD] url={url}")
        print(f"[WAN DOWNLOAD] saving to {output_path}")
        try:
            resp = requests.get(url, stream=True, timeout=(10, 300))
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (256 * 1024) == 0:
                        print(
                            f"[WAN DOWNLOAD] {downloaded / 1024:.0f}KB / {total / 1024:.0f}KB"
                        )
            print(
                f"[WAN DOWNLOAD] Done: {downloaded / 1024:.0f}KB saved to {output_path}"
            )
            return True
        except requests.RequestException as e:
            print(f"[WAN DOWNLOAD] FAILED: {e}")
            return False

    def build_prompt(
        self,
        product: str = "cup",
        branded: str = "Coca-Cola",
        item: str = "can",
        template: Optional[str] = None,
    ) -> str:
        tmpl = template or DEFAULT_PROMPT_TEMPLATE
        return tmpl.format(product=product, branded=branded, item=item)

    def _poll_loop(self):
        while self._polling:
            for task_id in list(self._pending.keys()):
                task = self.poll_task(task_id)

                if task.status in ("SUCCEEDED", "FAILED"):
                    self._pending.pop(task_id, None)
                    self._completed[task_id] = task

                    callback = self._callbacks.pop(task_id, None)
                    if callback:
                        try:
                            callback(task)
                        except Exception as e:
                            logger.error("Callback error for %s: %s", task_id, e)

            time.sleep(self.poll_interval)
