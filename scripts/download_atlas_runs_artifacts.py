from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request

from google.auth import default
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
BASE_URL = "https://www.googleapis.com/drive/v3/files"
OUT_ROOT = os.path.join(os.getcwd(), "colab", "colab_runs")
TARGET_FOLDER = "atlas_runs"



def drive_request(url: str, headers: dict, method: str = "GET", data: bytes | None = None):
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, context=ssl.create_default_context()) as resp:
        return resp.read()


def get_access_headers():
    creds, project = default(scopes=SCOPES)
    creds.refresh(Request())
    return {
        "Authorization": f"Bearer {creds.token}",
        "x-goog-user-project": project or "",
    }


def list_files(headers, parent_id: str):
    files = []
    page_token = None
    while True:
        params = {
            "q": f"'{parent_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType,size)",
            "pageSize": 100,
            "spaces": "drive",
        }
        if page_token:
            params["pageToken"] = page_token

        url = BASE_URL + "?" + urllib.parse.urlencode(params)
        payload = json.loads(drive_request(url, headers))
        files.extend(payload.get("files", []))

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return files


def find_folder(headers, name: str, parent_id: str | None = None):
    query = (
        f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    if parent_id:
        query += f" and '{parent_id}' in parents"

    params = {
        "q": query,
        "fields": "files(id,name)",
        "spaces": "drive",
    }
    payload = json.loads(drive_request(BASE_URL + "?" + urllib.parse.urlencode(params), headers))
    return payload.get("files", [])


def download_file(headers, file_id: str, local_path: str, size: int | None = None):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip if file already exists with matching size.
    if os.path.exists(local_path) and size is not None:
        try:
            if os.path.getsize(local_path) == size:
                return True, "skipped"
        except OSError:
            pass

    url = BASE_URL + f"/{file_id}?alt=media"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ssl.create_default_context()) as resp:
        with open(local_path, "wb") as f:
            f.write(resp.read())

    return True, "downloaded"


def mirror_folder(headers, folder_id: str, local_root: str):
    pending = [(folder_id, local_root)]
    while pending:
        current_id, current_local = pending.pop()
        os.makedirs(current_local, exist_ok=True)

        for item in list_files(headers, current_id):
            name = item["name"]
            item_id = item["id"]
            mime = item["mimeType"]
            is_folder = mime == "application/vnd.google-apps.folder"

            if is_folder:
                pending.append((item_id, os.path.join(current_local, name)))
                continue

            target = os.path.join(current_local, name)
            size = item.get("size")
            size_int = int(size) if size else None
            _, status = download_file(headers, item_id, target, size_int)
            print(f"{status}: {target}")


def main():
    headers = get_access_headers()

    # Find atlas_runs at My Drive root first.
    root_folders = find_folder(headers, TARGET_FOLDER, parent_id="root")

    if not root_folders:
        # Fallback any folder named atlas_runs.
        root_folders = find_folder(headers, TARGET_FOLDER, parent_id=None)

    if not root_folders:
        raise RuntimeError('Could not find Drive folder "atlas_runs"')

    atlas_runs = root_folders[0]
    folder_id = atlas_runs["id"]

    print(f"Found atlas_runs folder: {folder_id}")
    out_root = os.path.join(OUT_ROOT, TARGET_FOLDER)

    mirror_folder(headers, folder_id, out_root)
    print(f"Done. Artifacts mirrored to: {out_root}")


if __name__ == "__main__":
    main()
