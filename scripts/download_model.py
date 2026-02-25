"""Download best.pt from Google Drive using gcloud application-default credentials."""
import io, os, sys

from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Target path on Drive: atlas_runs/yolo26n_components_v1/weights/best.pt
DRIVE_PATH = ['atlas_runs', 'yolo26n_components_v1', 'weights', 'best.pt']
LOCAL_DEST = r'c:\new_atlas\tools\annotator\data\master\best.pt'

def main():
    # Authenticate using gcloud application-default credentials
    creds, project = default(scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    # Walk the folder path
    parent_id = 'root'
    for i, name in enumerate(DRIVE_PATH):
        is_file = (i == len(DRIVE_PATH) - 1)
        if is_file:
            q = f"name='{name}' and '{parent_id}' in parents and trashed=false"
        else:
            q = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        
        results = service.files().list(q=q, fields='files(id, name, modifiedTime, size)').execute()
        files = results.get('files', [])
        if not files:
            print(f"ERROR: '{name}' not found in Drive path: {'/'.join(DRIVE_PATH[:i+1])}")
            sys.exit(1)
        
        item = files[0]
        if not is_file:
            parent_id = item['id']
            print(f"  Found folder: {item['name']} (id={item['id']})")
        else:
            print(f"  Found file: {item['name']}")
            print(f"  Modified: {item.get('modifiedTime', 'unknown')}")
            print(f"  Size: {int(item.get('size', 0)) / 1024 / 1024:.1f} MB")
            
            # Download
            os.makedirs(os.path.dirname(LOCAL_DEST), exist_ok=True)
            request = service.files().get_media(fileId=item['id'])
            with open(LOCAL_DEST, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"  Download: {int(status.progress() * 100)}%")
            
            size_mb = os.path.getsize(LOCAL_DEST) / 1024 / 1024
            print(f"\n✓ Saved to: {LOCAL_DEST} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
