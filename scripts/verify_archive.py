#!/usr/bin/env python3
"""Check bundled research evidence and authored document links offline."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_file(relative):
    require(not Path(relative).is_absolute(), f'Absolute archive path: {relative}')
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f'Path escapes archive: {relative}')
    require(path.is_file(), f'Missing file: {relative}')
    return path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    manifest = json.loads((ROOT / 'data/evidence-manifest.json').read_text())
    copied = [e for e in manifest['entries'] if 'archive_path' in e]
    require(len(copied) == manifest['copied_files'], 'Evidence file count differs')
    require(len(manifest['entries']) - len(copied) == manifest['reference_only_entries'], 'Reference count differs')
    require(sum(e['archive_bytes'] for e in copied) == manifest['copied_bytes'], 'Evidence byte total differs')
    require(len({e['archive_path'] for e in copied}) == len(copied), 'Duplicate archive evidence paths')
    for entry in copied:
        path = local_file(entry['archive_path'])
        require(path.stat().st_size == entry['archive_bytes'], f'Size differs: {entry["archive_path"]}')
        require(digest(path) == entry['archive_sha256'], f'Digest differs: {entry["archive_path"]}')
        if entry['status'] == 'copied_unchanged':
            require(entry['original_sha256'] == entry['archive_sha256'], 'Unchanged-copy digest mismatch')
    inventory = json.loads((ROOT / 'data/archive-files.json').read_text())
    for entry in inventory['files']:
        path = local_file(entry['path'])
        require(path.stat().st_size == entry['bytes'], f'Published file size differs: {entry["path"]}')
        require(digest(path) == entry['sha256'], f'Published file digest differs: {entry["path"]}')
    source_map = json.loads((ROOT / 'data/source-map.json').read_text())
    require(source_map == {e['original_path']:e['archive_path'] for e in copied}, 'Source map differs')
    links = 0
    docs = [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]
    for document in docs:
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            target = target.strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith('#'):
                continue
            require(not parsed.path.startswith('/'), f'Nonportable link in {document.name}: {target}')
            path = (document.parent / unquote(parsed.path)).resolve()
            require(path.is_relative_to(ROOT) and path.exists(), f'Broken local link in {document.name}: {target}')
            links += 1
    return {'status':'verified','evidence_files':len(copied),
            'reference_only_entries':manifest['reference_only_entries'],
            'published_files_checked':len(inventory['files']),
            'authored_documents':len(docs),'local_document_links_checked':links,
            'evidence_bytes':manifest['copied_bytes'],
            'scope':'Local hashes, sizes, manifest accounting and authored-document links only; no inference or network.'}


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), indent=2))
    except Exception as error:
        print(json.dumps({'status':'failed','error':str(error)}), file=sys.stderr)
        raise SystemExit(1)
