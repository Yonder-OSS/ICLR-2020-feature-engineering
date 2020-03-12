import requests
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime

""" Setup """
ACCESS_TOKEN = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IlJqa3dNMEpFTURsRlFrSXdOemxDUlVZelJqQkdPRFpHUVRaRVFqWkRNRVJGUWpjeU5ERTFPQSJ9.eyJpc3MiOiJodHRwczovL3JhZGlhbnRlYXJ0aC5hdXRoMC5jb20vIiwic3ViIjoiYXV0aDB8NWU1ZDNkZGI1YjYwNmQwZDVlMmQxODFkIiwiYXVkIjpbImh0dHBzOi8vYXBpLnJhZGlhbnQuZWFydGgvdjEiLCJodHRwczovL3JhZGlhbnRlYXJ0aC5hdXRoMC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNTgzMTY4OTg4LCJleHAiOjE1ODM3NzM3ODgsImF6cCI6IlAzSXFMcWJYUm0xMEJVSk1IWEJVdGU2U0FEbjBTOERlIiwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCIsInBlcm1pc3Npb25zIjpbXX0.j2couzJ-bLyElKj_l3dduuijUCYETzPEKw6BtWjpmrKUdjudbRl62_heN43naembWUb4tn2i0G3bRNtGX_MrA7b9eNsmrIroB1QgaNvN390FKz-Bfkk3KP2smC3008zN5Iyr9-Iuf59VTBuDm6Tpov18KVFy_P8eGv_MB6XB_7OXsqOPLPpEaVw3TGFdJJvJaLODvWDdBejvGyLHomJlQ1SdwZUG9Z4X1eXsBI5yeRUrV7Xl3Ag5No4xhADSuou6AxPSpg-ZsqBAIcCeeX2P5NTofv76Z82PxA4bNTfLaQ9s5our9jnIbEyYMrchXJz5ZqpW9XXQoqpnepKHrKOu0g'
OUTPATH = '/home/snamjoshi/Documents/datasets/iclr_challenge_data/'

output_path = Path(OUTPATH)

headers = {
    'Authorization': f'Bearer {ACCESS_TOKEN}',
    'Accept':'application/json'
}

""" Functions """
def get_download_url(item, asset_key, headers):
    asset = item.get('assets', {}).get(asset_key, None)
    if asset is None:
        print(f'Asset "{asset_key}" does not exist in this item')
        return None
    r = requests.get(asset.get('href'), headers=headers, allow_redirects=False)
    return r.headers.get('Location')

def download_label(url, output_path, tileid):
    filename = urlparse(url).path.split('/')[-1]
    outpath = output_path/tileid
    outpath.mkdir(parents=True, exist_ok=True)

    r = requests.get(url)
    f = open(outpath/filename, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024):
        if chunk:
            f.write(chunk)
    f.close()
    print(f'Downloaded {filename}')
    return

def download_imagery(url, output_path, tileid, date):
    filename = urlparse(url).path.split('/')[-1]
    outpath = output_path/tileid/date
    outpath.mkdir(parents=True, exist_ok=True)

    r = requests.get(url)
    f = open(outpath/filename, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024):
        if chunk:
            f.write(chunk)
    f.close()
    print(f'Downloaded {filename}')
    return

""" Downloading labels """
# paste the id of the labels collection:
collectionId = 'ref_african_crops_kenya_02_labels'

# these optional parameters can be used to control what items are returned.
# Here, we want to download all the items so:
limit = 100
bounding_box = []
date_time = []

# retrieves the items and their metadata in the collection
r = requests.get(f'https://api.radiant.earth/mlhub/v1/collections/{collectionId}/items', params={'limit':limit, 'bbox':bounding_box,'datetime':date_time}, headers=headers)
collection = r.json()

for feature in collection.get('features', []):
    assets = feature.get('assets').keys()
    print("Feature", feature.get('id'), 'with the following assets', list(assets))

for feature in collection.get('features', []):

    tileid = feature.get('id').split('tile_')[-1][:2]

    # download labels
    download_url = get_download_url(feature, 'labels', headers)
    download_label(download_url, output_path, tileid)

    #download field_ids
    download_url = get_download_url(feature, 'field_ids', headers)
    download_label(download_url, output_path, tileid)

""" Downloading imagery """
# This cell downloads all the multi-spectral images throughout the growing season for this competition.
# The size of data is about 1.5 GB, and download time depends on your internet connection.
# Note that you only need to run this cell and download the data once.
for feature in collection.get('features', []):
    for link in feature.get('links', []):
        if link.get('rel') != 'source':
            continue

        r = requests.get(link['href'], headers=headers)
        feature = r.json()
        assets = feature.get('assets').keys()
        tileid = feature.get('id').split('tile_')[-1][:2]
        date = datetime.strftime(datetime.strptime(feature.get('properties')['datetime'], "%Y-%m-%dT%H:%M:%SZ"), "%Y%m%d")
        for asset in assets:
            download_url = get_download_url(feature, asset, headers)
            download_imagery(download_url, output_path, tileid, date)

