"""Check Google Drive for ATLAS training files."""
import json, urllib.request, urllib.parse, ssl

# Use Application Default Credentials (with Drive scope)
import google.auth
import google.auth.transport.requests

creds, project = google.auth.default(
    scopes=['https://www.googleapis.com/auth/drive.readonly']
)
creds.refresh(google.auth.transport.requests.Request())
token = creds.token

headers = {
    'Authorization': f'Bearer {token}',
    'x-goog-user-project': project or 'gen-lang-client-0746582623',
}
ctx = ssl.create_default_context()

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

def list_children(folder_id, indent=0):
    page_token = None
    while True:
        params = {
            'q': f"'{folder_id}' in parents and trashed=false",
            'fields': 'nextPageToken,files(id,name,mimeType,size)',
            'pageSize': 100,
            'spaces': 'drive',
        }
        if page_token:
            params['pageToken'] = page_token
        result = drive_api('files', params)
        for f in sorted(result.get('files', []), key=lambda x: x['name']):
            is_folder = f['mimeType'] == 'application/vnd.google-apps.folder'
            size = f.get('size', '')
            if size and not is_folder:
                size_mb = int(size) / 1024 / 1024
                size_str = f' ({size_mb:.1f} MB)' if size_mb >= 1 else f' ({int(size) / 1024:.0f} KB)'
            else:
                size_str = ''
            prefix = '  ' * indent
            marker = '/' if is_folder else ''
            print(f'{prefix}{f["name"]}{marker}{size_str}')
            if is_folder:
                list_children(f['id'], indent + 1)
        page_token = result.get('nextPageToken')
        if not page_token:
            break

# Look for Atlas folder
print('=== Searching for "Atlas" folder ===')
atlas_folders = find_folder('Atlas')
if not atlas_folders:
    print('No "Atlas" folder found in Drive root.')
    # Try searching anywhere
    q = "name='Atlas' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    result = drive_api('files', {'q': q, 'fields': 'files(id,name,parents)', 'spaces': 'drive'})
    atlas_folders = result.get('files', [])
    if atlas_folders:
        print(f'Found {len(atlas_folders)} "Atlas" folder(s) elsewhere:')

for folder in atlas_folders:
    print(f'\nAtlas folder ID: {folder["id"]}')
    list_children(folder['id'])

# Look for atlas_runs folder
print('\n=== Searching for "atlas_runs" folder ===')
runs_folders = find_folder('atlas_runs')
if not runs_folders:
    q = "name='atlas_runs' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    result = drive_api('files', {'q': q, 'fields': 'files(id,name)', 'spaces': 'drive'})
    runs_folders = result.get('files', [])

for folder in runs_folders:
    print(f'\natlas_runs folder ID: {folder["id"]}')
    list_children(folder['id'])

if not atlas_folders and not runs_folders:
    print('\nNo Atlas or atlas_runs folders found. Listing My Drive root...')
    root = drive_api('files', {
        'q': "'root' in parents and trashed=false",
        'fields': 'files(id,name,mimeType)',
        'pageSize': 50,
        'spaces': 'drive',
    })
    for f in sorted(root.get('files', []), key=lambda x: x['name']):
        marker = '/' if f['mimeType'] == 'application/vnd.google-apps.folder' else ''
        print(f'  {f["name"]}{marker}')
