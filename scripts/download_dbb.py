#!/usr/bin/env python3
"""Scarica il progetto brainlife "DBB Distorted Brain Benchmark" in data/raw/DBB.

DBB (brainlife project 60a14ca503bcad0ad27cada9, DOI 10.25663/brainlife.pub.24,
licenza CC-BY 4.0) e' un benchmark di segmentazione dei *tessuti* cerebrali su
cervelli pediatrici malformati. Non contiene label di lesione.

Credenziali: variabili d'ambiente `BRAINLIFE_USER` e `BRAINLIFE_PASSWORD`, oppure
le stesse chiavi in `.env` (che e' escluso da Git). Non vanno passate a riga di
comando: finirebbero nella history della shell.

Ogni dataset brainlife arriva come un `.tar` che viene estratto in
    data/raw/DBB/sub-<subject>/<datatype-slug>/<dataset_id>/
Il download e' ripartibile: le cartelle che contengono gia' il marker `.done`
vengono saltate, quindi rilanciare lo script completa un trasferimento interrotto.

Uso:
    python scripts/download_dbb.py [--workers 6]
"""
import argparse
import csv
import io
import json
import os
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'raw' / 'DBB'
PROJECT = '60a14ca503bcad0ad27cada9'
AUTH_URL = 'https://brainlife.io/api/auth/local/auth'
API = 'https://brainlife.io/api/warehouse'
RETRIES = 4

DATATYPE_SLUG = {
    '58c33bcee13a50849b25879a': 'anat-t1w',
    '5a281aee2c214c9ba83ce620': 'mask',
    '5bbfb28071454db2a890fbce': 'transform-nifti',
    '5c1a7489f9109beac4a88a1f': 'parcellation-volume',
}

lock = threading.Lock()
state = {'ok': 0, 'skip': 0, 'fail': 0, 'bytes': 0}
failures = []


def credentials():
    env_file = ROOT / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())
    user = os.environ.get('BRAINLIFE_USER')
    password = os.environ.get('BRAINLIFE_PASSWORD')
    if not user or not password:
        sys.exit('Servono BRAINLIFE_USER e BRAINLIFE_PASSWORD (ambiente o .env).')
    return user, password


def sign_in(user, password):
    body = json.dumps({'username': user, 'password': password}).encode()
    req = urllib.request.Request(AUTH_URL, data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)['jwt']


def api_get(jwt, path, params):
    url = API + path + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + jwt})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def list_datasets(jwt):
    """Elenca i dataset del progetto, paginando con ordinamento **stabile**.

    Senza `sort` l'API non garantisce lo stesso ordine fra una pagina e l'altra:
    alcuni record tornano due volte e altrettanti non compaiono mai. L'ordinamento
    per `_id` rende la paginazione deterministica; la deduplica e' una rete di
    sicurezza in piu'.
    """
    found = {}
    skip = 0
    while True:
        page = api_get(jwt, '/dataset', {
            'find': json.dumps({'project': PROJECT, 'removed': False}),
            'limit': 500, 'skip': skip, 'sort': '_id',
        }).get('datasets', [])
        for rec in page:
            found[rec['_id']] = rec
        if len(page) < 500:
            break
        skip += 500
    return [found[k] for k in sorted(found)]


def slug_for(rec):
    base = DATATYPE_SLUG.get(rec.get('datatype'), rec.get('datatype', 'unknown'))
    tags = [t for t in (rec.get('datatype_tags') or []) if t]
    return base + ('_' + '-'.join(sorted(tags)) if tags else '')


def target_dir(rec):
    subject = (rec.get('meta') or {}).get('subject') or 'unknown'
    return DEST / ('sub-' + str(subject)) / slug_for(rec) / rec['_id']


def fetch(jwt, rec):
    out = target_dir(rec)
    if (out / '.done').exists():
        with lock:
            state['skip'] += 1
        return
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(API + '/dataset/download/' + rec['_id'],
                                         headers={'Authorization': 'Bearer ' + jwt})
            with urllib.request.urlopen(req, timeout=300) as r:
                blob = r.read()
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
                tf.extractall(out)
            (out / '.done').touch()
            with lock:
                state['ok'] += 1
                state['bytes'] += len(blob)
            return
        except Exception as exc:  # rete, 5xx, tar troncato
            last = exc
            time.sleep(2 * (attempt + 1))
    with lock:
        state['fail'] += 1
        failures.append((rec['_id'], repr(last)))


def write_manifest(records):
    with (DEST / 'manifest.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['dataset_id', 'subject', 'datatype', 'datatype_slug',
                         'tags', 'size_bytes', 'path'])
        for rec in records:
            writer.writerow([
                rec['_id'],
                (rec.get('meta') or {}).get('subject'),
                rec.get('datatype'),
                slug_for(rec),
                '|'.join(rec.get('tags') or []),
                rec.get('size') or 0,
                str(target_dir(rec).relative_to(DEST)),
            ])
    (DEST / 'manifest.json').write_text(json.dumps(records))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    jwt = sign_in(*credentials())
    records = list_datasets(jwt)
    DEST.mkdir(parents=True, exist_ok=True)
    write_manifest(records)
    print('dataset da scaricare: %d' % len(records), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(lambda r: fetch(jwt, r), records), 1):
            if i % 200 == 0:
                print('%d/%d  ok=%d skip=%d fail=%d  %.2f GB' % (
                    i, len(records), state['ok'], state['skip'], state['fail'],
                    state['bytes'] / 1e9), flush=True)

    print('FINE  ok=%d skip=%d fail=%d  %.2f GB' % (
        state['ok'], state['skip'], state['fail'], state['bytes'] / 1e9), flush=True)
    for dataset_id, err in failures[:10]:
        print('  fallito', dataset_id, err, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
