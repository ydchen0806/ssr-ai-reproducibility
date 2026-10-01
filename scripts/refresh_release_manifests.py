"""Refresh release checksums after an intentional source or code update."""

import csv
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Data_S1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_data():
    path = DATA / 'MANIFEST.csv'
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    assert fields and len(rows) == len({row['file'] for row in rows})
    actual = {str(file.relative_to(DATA)) for file in DATA.rglob('*') if file.is_file() and file != path}
    assert {row['file'] for row in rows} == actual, 'Data S1 manifest inventory differs from delivered files'
    for row in rows:
        row['delivered_sha256'] = sha256(DATA / row['file'])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def refresh_repository():
    names = subprocess.check_output(
        ['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT,
    ).decode().split('\0')
    paths = sorted({name for name in names if name and name != 'FILE_MANIFEST.csv' and (ROOT / name).is_file()})
    with (ROOT / 'FILE_MANIFEST.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['path', 'bytes', 'sha256', 'storage'])
        for name in paths:
            file = ROOT / name
            writer.writerow([name, file.stat().st_size, sha256(file), 'git'])
    return len(paths)


if __name__ == '__main__':
    print({'data_s1_files': refresh_data(), 'repository_files': refresh_repository()})
