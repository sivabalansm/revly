"""
Upload video clips to public temporary hosting for Wan 2.7 API consumption.
Fallback chain: tmpfiles.org → litterbox.catbox.moe → transfer.sh
"""

import os
import logging
import time

import requests

logger = logging.getLogger("adstream.video_uploader")


def upload_video(file_path: str, timeout: int = 120) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    logger.info("Uploading %s (%.1f KB)", file_path, file_size / 1024)

    errors = []

    for upload_fn in [_upload_tmpfiles, _upload_litterbox, _upload_transfersh]:
        try:
            url = upload_fn(file_path, timeout)
            if url:
                logger.info("Upload success: %s", url)
                return url
        except Exception as e:
            name = upload_fn.__name__.replace("_upload_", "")
            errors.append(f"{name}: {e}")
            logger.warning("Upload to %s failed: %s", name, e)

    raise RuntimeError(f"All upload services failed: {'; '.join(errors)}")


def _upload_tmpfiles(file_path: str, timeout: int) -> str:
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": f},
            timeout=timeout,
        )
    resp.raise_for_status()
    data = resp.json()

    view_url = data["data"]["url"]
    direct_url = view_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    return direct_url


def _upload_litterbox(file_path: str, timeout: int) -> str:
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://litterbox.catbox.moe/resources/internals/api.php",
            data={"reqtype": "fileupload", "time": "1h"},
            files={"fileToUpload": f},
            timeout=timeout,
        )
    resp.raise_for_status()
    url = resp.text.strip()

    if not url.startswith("http"):
        raise ValueError(f"Invalid response: {url[:100]}")
    return url


def _upload_transfersh(file_path: str, timeout: int) -> str:
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        resp = requests.put(
            f"https://transfer.sh/{filename}",
            data=f,
            timeout=timeout,
        )
    resp.raise_for_status()
    url = resp.text.strip()

    if not url.startswith("http"):
        raise ValueError(f"Invalid response: {url[:100]}")
    return url
