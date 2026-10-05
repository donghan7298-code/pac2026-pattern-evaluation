import gzip
import hashlib
import json
from pathlib import Path


def stable_seed(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def opaque_id(*parts):
    return hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:20]


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class JsonlWriter:
    """Deterministic gzip header; identical input produces identical compressed bytes."""
    def __init__(self, path):
        self.raw = open(path, 'wb')
        self.gz = gzip.GzipFile(fileobj=self.raw, mode='wb', filename='', mtime=0, compresslevel=6)

    def write(self, value):
        self.gz.write((json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode())

    def close(self):
        self.gz.close()
        self.raw.close()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
