#!/usr/bin/env python
"""Eta' dei 17 pazienti AInCP, dai fogli di metadati DICOM dei due file di elenco.

    python scripts/eta_aincp.py     # -> reports/eta_aincp.md

PERCHE'. Le istruzioni della tesi dichiarano l'eta' del campione (3–15 anni, quattro pazienti sotto i 5
anni). Quel numero va verificato sui dati, non ripetuto.

COME. L'eta' e' il campo DICOM «Patient's Age» (formato NNNY, anni compiuti), presente nei fogli `FSM`
e `Sheet1` (v2) e `FSM` (v1) dei due workbook in docs/. Per ogni paziente della pipeline
(pipeline_aincp.trova_casi) si cerca la voce in tutti i fogli; se due fogli la danno, devono coincidere.
Il foglio `SITE & MRFIELD` ha la stessa informazione in «AGE AT SCAN» (solo alcuni pazienti). Un
paziente senza nessuna voce viene segnalato, non stimato.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_aincp as pa  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
DOCS = PROGETTO / 'docs'
FILE = {'v1': DOCS / 'AINCP_SERIES_TYPE_dataset_FSM_v1.xlsx', 'v2': DOCS / 'AINCP_SERIES_TYPE_dataset_FSM_v2.xlsx'}
FOGLI = [('v2', 'Sheet1', 'Patient’s Age', 'PatientID'), ('v2', 'FSM', 'Patient’s Age', 'PatientID'),
         ('v1', 'FSM', 'Patient’s Age', 'PatientID'), ('v1', 'SITE & MRFIELD', 'AGE AT SCAN', 'SESSION'),
         ('v2', 'SITE & MRFIELD', 'AGE AT SCAN', 'SESSION')]


def anni(v):
    """'007Y' -> 7; altri formati (settimane, mesi, giorni) non ce ne sono per questi pazienti: si segnalano."""
    m = re.fullmatch(r'0*(\d+)Y', str(v).strip().upper())
    return int(m.group(1)) if m else None


def leggi():
    voci = {}                                              # paziente -> [(file, foglio, valore grezzo)]
    for ver, foglio, colonna, chiave in FOGLI:
        try:
            d = pd.read_excel(FILE[ver], foglio)
        except (ValueError, KeyError):
            continue
        if colonna not in d.columns or chiave not in d.columns:
            continue
        for _, r in d.dropna(subset=[colonna]).iterrows():
            k = str(r[chiave]).split('_MR_')[0].strip()    # 'FSM_001_MR_2' -> 'FSM_001'
            if k.startswith('FSM_'):
                voci.setdefault(k, []).append((ver, foglio, str(r[colonna]).strip()))
    return voci


def main():
    casi = sorted(pa.trova_casi())
    voci = leggi()
    righe, problemi = [], []
    for p in casi:
        v = voci.get(p, [])
        valori = {anni(x[2]) for x in v}
        if not v:
            righe.append((p, '—', 'nessuna voce', ''))
            problemi.append(f'{p}: nessuna età nei fogli')
            continue
        if None in valori:
            problemi.append(f'{p}: formato non riconosciuto {v}')
        if len(valori - {None}) > 1:
            problemi.append(f'{p}: fogli in disaccordo {v}')
        eta = min(valori - {None})
        fonti = '; '.join(sorted({f'{ver} {f} «{raw}»' for ver, f, raw in v}))
        righe.append((p, eta, fonti, 'sì' if eta < 5 else ('5 anni compiuti' if eta == 5 else 'no')))
    sotto5 = [p for p, e, _, _ in righe if isinstance(e, int) and e < 5]
    cinque = [p for p, e, _, _ in righe if e == 5]
    eta_v = [e for _, e, _, _ in righe if isinstance(e, int)]
    md = ['# Età dei pazienti AInCP (17)', '',
          'Prodotto da `scripts/eta_aincp.py`, **non modificare a mano**. Fonte: campo DICOM «Patient\'s Age» '
          '(anni compiuti) nei fogli di metadati dei due file in `docs/`.', '',
          f'Età: da {min(eta_v)} a {max(eta_v)} anni; mediana {sorted(eta_v)[len(eta_v) // 2]}. '
          f'Sotto i 5 anni (strettamente): **{len(sotto5)}** ({", ".join(sotto5)}). '
          f'A esattamente 5 anni: **{len(cinque)}** ({", ".join(cinque)}). '
          f'Oltre i 5 anni: **{len(eta_v) - len(sotto5) - len(cinque)}**.', '',
          '| paziente | età (anni) | fonte (file, foglio, valore) | sotto 5 anni |', '| --- | --- | --- | --- |']
    md += [f'| {p} | {e} | {f} | {s} |' for p, e, f, s in righe]
    md += ['', 'Controlli: ' + ('nessun problema.' if not problemi else '; '.join(problemi) + '.')]
    (PROGETTO / 'reports/eta_aincp.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
