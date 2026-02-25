"""Upload atlas-components dataset to Google Drive at Atlas/train/.

Replaces existing contents of Atlas/train/ with the new component dataset.
"""
import json, os, urllib.request, urllib.parse, ssl, mimetypes

import google.auth
import google.auth.transport.requests

# Authenticate
creds, project = google.auth.default(
    scopes=['https://www.googleapis.com/auth/drive']
)
creds.refresh(google.auth.transport.requests.Request())
token = creds.token

headers = {
    'Authorization': f'Bearer {token}',
    'x-goog-user-project': project or 'gen-lang-client-0746582623',
}
ctx = ssl.create_default_context()

LOCAL_DATASET = r'c:\new_atlas\atlas-components'


def drive_api(endpoint, params=None):
    url = f'https://www.googleapis.com/drive/v3/{endpoint}'
    if params:
        url += '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx) as resp:
        return json.loads(resp.read())


def find_folder(name, parent_id=None):
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    result = drive_api('files', {'q': q, 'fields': 'files(id,name)', 'spaces': 'drive'})
    return result.get('files', [])


def create_folder(name, parent_id=None):
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        meta['parents'] = [parent_id]
    body = json.dumps(meta).encode()
    req = urllib.request.Request(
        'https://www.googleapis.com/drive/v3/files',
        data=body,
        headers={**headers, 'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, context=ctx) as resp:
        return json.loads(resp.read())


def delete_file(file_id):
    req = urllib.request.Request(
        f'https://www.googleapis.com/drive/v3/files/{file_id}',
        headers=headers,
        method='DELETE'
    )
    urllib.request.urlopen(req, context=ctx)


def list_children_ids(folder_id):
    """Get all children (files + folders) of a folder."""
    items = []
    page_token = None
    while True:
        params = {
            'q': f"'{folder_id}' in parents and trashed=false",
            'fields': 'nextPageToken,files(id,name,mimeType)',
            'pageSize': 100,
            'spaces': 'drive',
        }
        if page_token:
            params['pageToken'] = page_token
        result = drive_api('files', params)
        items.extend(result.get('files', []))
        page_token = result.get('nextPageToken')
        if not page_token:
            break
    return items


def upload_file(local_path, parent_id):
    """Upload a file using multipart upload."""
    filename = os.path.basename(local_path)
    mime_type = mimetypes.guess_type(local_path)[0] or 'application/octet-stream'

    meta = json.dumps({'name': filename, 'parents': [parent_id]}).encode()

    with open(local_path, 'rb') as f:
        file_data = f.read()

    boundary = b'----UploadBoundary12345'
    body = (
        b'--' + boundary + b'\r\n'
        b'Content-Type: application/json; charset=UTF-8\r\n\r\n'
        + meta + b'\r\n'
        b'--' + boundary + b'\r\n'
        b'Content-Type: ' + mime_type.encode() + b'\r\n\r\n'
        + file_data + b'\r\n'
        b'--' + boundary + b'--'
    )

    req = urllib.request.Request(
        'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart',
        data=body,
        headers={
            **headers,
            'Content-Type': f'multipart/related; boundary={boundary.decode()}',
            'Content-Length': str(len(body)),
        },
        method='POST'
    )
    with urllib.request.urlopen(req, context=ctx) as resp:
        return json.loads(resp.read())


def clear_folder(folder_id):
    """Delete all contents of a folder."""
    children = list_children_ids(folder_id)
    for child in children:
        if child['mimeType'] == 'application/vnd.google-apps.folder':
            clear_folder(child['id'])
        print(f'  Deleting {child["name"]}...')
        delete_file(child['id'])


def ensure_folder(name, parent_id):
    """Find or create a folder."""
    existing = find_folder(name, parent_id)
    if existing:
        return existing[0]['id']
    result = create_folder(name, parent_id)
    return result['id']


# --- Main ---
print('=== Finding Atlas/train folder ===')
atlas_folders = find_folder('Atlas')
if not atlas_folders:
    print('Creating Atlas folder...')
    atlas = create_folder('Atlas')
    atlas_id = atlas['id']
else:
    atlas_id = atlas_folders[0]['id']
    print(f'Atlas folder: {atlas_id}')

train_folders = find_folder('train', atlas_id)
if train_folders:
    train_id = train_folders[0]['id']
    print(f'Found existing train folder: {train_id}')
    print('Clearing old contents...')
    clear_folder(train_id)
else:
    result = create_folder('train', atlas_id)
    train_id = result['id']
    print(f'Created train folder: {train_id}')

# Upload dataset structure
print('\n=== Uploading dataset ===')
for root, dirs, files in os.walk(LOCAL_DATASET):
    # Determine relative path from dataset root
    rel = os.path.relpath(root, LOCAL_DATASET)
    if rel == '.':
        parent_id = train_id
    else:
        # Navigate/create folder path
        parent_id = train_id
        for part in rel.split(os.sep):
            parent_id = ensure_folder(part, parent_id)

    for fname in sorted(files):
        local_path = os.path.join(root, fname)
        size_kb = os.path.getsize(local_path) / 1024
        print(f'  Uploading {os.path.join(rel, fname)} ({size_kb:.0f} KB)...')
        upload_file(local_path, parent_id)

print('\n=== Upload complete! ===')
print(f'Dataset uploaded to: My Drive/Atlas/train/')
print(f'  data.yaml + train/ + valid/')
