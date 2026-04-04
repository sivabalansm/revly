"""
Live ad pipeline orchestrator.
YOLO detects → extract clip → upload → Wan 2.7 → poll → download → splice into buffer.
"""

import os
import time
import threading
import tempfile
import logging
from typing import Optional
from dataclasses import dataclass, field

import cv2
import numpy as np

from lib.wan_client import WanClient, WanTask
from lib.video_uploader import upload_video
from lib.stream_buffer import StreamBuffer

logger = logging.getLogger("adstream.ad_pipeline")


@dataclass
class AdJob:
    job_id: str
    segment_start_ts: float
    segment_end_ts: float
    wan_task: Optional[WanTask] = None
    clip_path: Optional[str] = None
    clip_url: Optional[str] = None
    edited_path: Optional[str] = None
    status: str = "extracting"
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    error: Optional[str] = None


class LiveAdPipeline:
    def __init__(
        self,
        wan_client: WanClient,
        stream_buffer: StreamBuffer,
        prompt_template: Optional[str] = None,
        product: str = "cup",
        branded: str = "Coca-Cola",
        item: str = "can",
        segment_duration: float = 5.0,
        segment_pre_pad: float = 0.5,
        max_concurrent_jobs: int = 3,
        temp_dir: Optional[str] = None,
    ):
        self.wan_client = wan_client
        self.stream_buffer = stream_buffer
        self.product = product
        self.branded = branded
        self.item = item
        self.segment_duration = segment_duration
        self.segment_pre_pad = segment_pre_pad
        self.max_concurrent_jobs = max_concurrent_jobs
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="adstream_pipeline_")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.prompt = wan_client.build_prompt(
            product=product,
            branded=branded,
            item=item,
            template=prompt_template,
        )

        self._jobs: dict[str, AdJob] = {}
        self._lock = threading.Lock()
        self._job_counter = 0
        self._cooldown_until = 0.0
        self._min_job_interval = 30.0

    def trigger_ad_replacement(
        self,
        detection_ts: float,
        reference_image_url: Optional[str] = None,
    ) -> Optional[AdJob]:
        now = time.time()
        if now < self._cooldown_until:
            logger.debug(
                "Ad pipeline on cooldown (%.1fs remaining)", self._cooldown_until - now
            )
            return None

        with self._lock:
            active = sum(
                1 for j in self._jobs.values() if j.status not in ("done", "failed")
            )
            if active >= self.max_concurrent_jobs:
                logger.warning(
                    "Max concurrent jobs reached (%d)", self.max_concurrent_jobs
                )
                return None

        end_ts = detection_ts
        start_ts = end_ts - self.segment_duration

        self._job_counter += 1
        job_id = f"ad_{self._job_counter}_{int(detection_ts)}"
        job = AdJob(job_id=job_id, segment_start_ts=start_ts, segment_end_ts=end_ts)

        with self._lock:
            self._jobs[job_id] = job

        self._cooldown_until = now + self._min_job_interval

        thread = threading.Thread(
            target=self._process_job,
            args=(job, reference_image_url),
            daemon=True,
        )
        thread.start()

        logger.info("Ad job started: %s [%.1f → %.1f]", job_id, start_ts, end_ts)
        return job

    def get_active_jobs(self) -> list[AdJob]:
        with self._lock:
            return [
                j for j in self._jobs.values() if j.status not in ("done", "failed")
            ]

    def get_completed_jobs(self) -> list[AdJob]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status == "done"]

    def cleanup_old_jobs(self, max_age: float = 300.0):
        now = time.time()
        with self._lock:
            stale = [
                jid
                for jid, j in self._jobs.items()
                if j.status in ("done", "failed") and now - j.created_at > max_age
            ]
            for jid in stale:
                job = self._jobs.pop(jid)
                if job.clip_path and os.path.exists(job.clip_path):
                    os.unlink(job.clip_path)
                if job.edited_path and os.path.exists(job.edited_path):
                    os.unlink(job.edited_path)

    def _process_job(self, job: AdJob, reference_image_url: Optional[str]):
        try:
            job.status = "extracting"
            print(
                f"  [{job.job_id}] Extracting {job.segment_end_ts - job.segment_start_ts:.1f}s clip..."
            )
            clip_path = self.stream_buffer.extract_segment_as_mp4(
                job.segment_start_ts,
                job.segment_end_ts,
                output_path=os.path.join(self.temp_dir, f"{job.job_id}_input.mp4"),
            )
            if not clip_path:
                job.status = "failed"
                job.error = "No frames in buffer for segment"
                print(f"  [{job.job_id}] FAILED: {job.error}")
                return
            file_size = os.path.getsize(clip_path)
            print(f"  [{job.job_id}] Extracted: {clip_path} ({file_size} bytes)")
            job.clip_path = clip_path

            job.status = "uploading"
            print(f"  [{job.job_id}] Uploading...")
            try:
                clip_url = upload_video(clip_path)
            except RuntimeError as e:
                job.status = "failed"
                job.error = f"Upload failed: {e}"
                print(f"  [{job.job_id}] FAILED: {job.error}")
                return
            job.clip_url = clip_url
            print(f"  [{job.job_id}] Uploaded: {clip_url}")

            job.status = "generating"
            print(f"  [{job.job_id}] Submitting to Wan 2.7...")
            ref_images = [reference_image_url] if reference_image_url else None
            wan_task = self.wan_client.submit(
                video_url=clip_url,
                prompt=self.prompt,
                reference_images=ref_images,
                metadata={"job_id": job.job_id},
            )
            if not wan_task:
                job.status = "failed"
                job.error = "Wan API submission failed"
                print(f"  [{job.job_id}] FAILED: {job.error}")
                return
            job.wan_task = wan_task
            print(f"  [{job.job_id}] Task ID: {wan_task.task_id}")

            job.status = "polling"
            print(
                f"  [{job.job_id}] Polling every {self.wan_client.poll_interval:.0f}s..."
            )
            while True:
                task = self.wan_client.poll_task(wan_task.task_id)
                job.wan_task = task

                if task.status == "SUCCEEDED":
                    break
                elif task.status in ("FAILED", "CANCELED"):
                    job.status = "failed"
                    job.error = f"Wan task {task.status}: {task.error}"
                    print(f"  [{job.job_id}] FAILED: {job.error}")
                    return

                time.sleep(self.wan_client.poll_interval)

            job.status = "downloading"
            print(f"  [{job.job_id}] Downloading edited video...")
            edited_path = os.path.join(self.temp_dir, f"{job.job_id}_edited.mp4")
            if not self.wan_client.download_video(task.video_url, edited_path):
                job.status = "failed"
                job.error = "Download failed"
                print(f"  [{job.job_id}] FAILED: {job.error}")
                return
            job.edited_path = edited_path
            print(f"  [{job.job_id}] Downloaded: {edited_path}")

            job.status = "splicing"
            print(f"  [{job.job_id}] Splicing into buffer...")
            replaced = self.stream_buffer.replace_segment_from_mp4(
                edited_path,
                job.segment_start_ts,
            )

            job.status = "done"
            job.completed_at = time.time()
            elapsed = job.completed_at - job.created_at
            logger.info(
                "Ad job complete: %s — %d frames replaced in %.1fs",
                job.job_id,
                replaced,
                elapsed,
            )

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("Ad job %s failed: %s", job.job_id, e)
