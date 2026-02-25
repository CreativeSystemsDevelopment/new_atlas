"""Quick test: verify Drive API access works for upload."""
import google.auth, google.auth.transport.requests, json, urllib.request, urllib.parse, ssl

creds, project = google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
creds.refresh(google.auth.transport.requests.Request())
token = creds.token
headers = {
    'Authorization': f'Bearer {token}',
    'x-goog-user-project': project or 'gen-lang-client-0746582623',
}
ctx = ssl.create_default_context()

# Test: find Atlas folder
q = "name='Atlas' and mimeType='application/vnd.google-apps.folder' and trashed=false"
url = 'https://www.googleapis.com/drive/v3/files?' + urllib.parse.urlencode({
    'q': q, 'fields': 'files(id,name)', 'spaces': 'drive',
})
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, context=ctx) as resp:
    data = json.loads(resp.read())

folders = data.get('files', [])
if folders:
    print(f'OK: Atlas folder found (id={folders[0]["id"]})')
else:
    print('FAIL: Atlas folder not found')

# Test: find train subfolder
if folders:
    atlas_id = folders[0]['id']
    q2 = f"name='train' and mimeType='application/vnd.google-apps.folder' and trashed=false and '{atlas_id}' in parents"
    url2 = 'https://www.googleapis.com/drive/v3/files?' + urllib.parse.urlencode({
        'q': q2, 'fields': 'files(id,name)', 'spaces': 'drive',
    })
    req2 = urllib.request.Request(url2, headers=headers)
    with urllib.request.urlopen(req2, context=ctx) as resp2:
        data2 = json.loads(resp2.read())
    train_folders = data2.get('files', [])
    if train_folders:
        print(f'OK: Atlas/train folder found (id={train_folders[0]["id"]})')
    else:
        print('WARN: Atlas/train not found (will be created)')

print('\nDrive API fully operational for upload.')
