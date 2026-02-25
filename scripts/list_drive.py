from google.auth import default
from googleapiclient.discovery import build
creds, _ = default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
svc = build('drive', 'v3', credentials=creds)
# Find atlas_runs folder
q1 = "name='atlas_runs' and 'root' in parents and trashed=false"
r = svc.files().list(q=q1, fields='files(id)').execute()
fid = r['files'][0]['id']
# List contents
q2 = f"'{fid}' in parents and trashed=false"
r2 = svc.files().list(q=q2, fields='files(id,name,modifiedTime)', pageSize=50).execute()
for f in sorted(r2['files'], key=lambda x: x['name']):
    print(f"{f['name']:40s}  {f.get('modifiedTime','')}")
