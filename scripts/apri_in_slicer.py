"""Apre in 3D Slicer un paziente esportato da esporta_slicer_506.py, con le viste gia' impostate.

Si esegue DENTRO Slicer (versione 5.x), in uno di questi due modi:

  1. dalla console Python di Slicer (menu View > Python Console):
        exec(open(r"C:/percorso/506_test/apri_in_slicer.py").read())
        apri(r"C:/percorso/506_test/FSM_046")            # ricetta Z, il modello della tesi
        apri(r"C:/percorso/506_test/FSM_046", "D")       # oppure A o D, per confronto

  2. dal terminale, aprendo Slicer gia' sul paziente:
        Slicer --python-script apri_in_slicer.py FSM_046 [Z|A|D]

Che cosa fa: svuota la scena, carica T1 e FLAIR, carica le tre segmentazioni di confronto (solo
quella scelta resta visibile; le altre si accendono dal modulo Data), costruisce le superfici 3D,
mette il cervello in trasparenza, imposta la vista a quattro riquadri (tre fette + 3D) e porta le
fette sulla lesione.

Colori dei segmenti: verde acqua = lesione trovata (annotata e predetta), blu = lesione mancata
(solo nell'annotazione), arancio = falso positivo (solo nella predizione).
"""
import os
import sys

import slicer

RICETTE = ('Z', 'A', 'D')


def _carica_segmentazione(percorso, nome, smussatura='0.2'):
    seg = slicer.util.loadSegmentation(percorso)
    seg.SetName(nome)
    # lo smussamento di default (0.5) fa sparire in 3D le lesioni di pochi voxel
    seg.GetSegmentation().SetConversionParameter('Smoothing factor', smussatura)
    seg.CreateClosedSurfaceRepresentation()
    return seg


def apri(cartella, ricetta='Z'):
    cartella = os.path.abspath(cartella)
    if ricetta not in RICETTE:
        raise ValueError(f'ricetta deve essere una fra {RICETTE}')
    slicer.mrmlScene.Clear(0)
    t1 = slicer.util.loadVolume(os.path.join(cartella, 'T1.nii.gz'), {'name': 'T1'})
    fl = slicer.util.loadVolume(os.path.join(cartella, 'FLAIR.nii.gz'), {'name': 'FLAIR'})

    cer = _carica_segmentazione(os.path.join(cartella, 'cervello.seg.nrrd'), 'cervello', '0.5')
    dc = cer.GetDisplayNode()
    dc.SetOpacity3D(0.12)
    dc.SetVisibility2D(False)

    scelta = None
    for k in RICETTE:
        f = os.path.join(cartella, f'confronto_{k}.seg.nrrd')
        if not os.path.exists(f):
            continue
        seg = _carica_segmentazione(f, f'confronto {k}')
        d = seg.GetDisplayNode()
        d.SetOpacity3D(0.9)
        d.SetOpacity2DFill(0.35)
        d.SetOpacity2DOutline(1.0)
        d.SetVisibility(k == ricetta)
        if k == ricetta:
            scelta = seg

    # sfondo FLAIR (la gliosi si vede li'); la T1 si sceglie dal menu della vista a fette
    slicer.util.setSliceViewerLayers(background=fl, fit=True)
    lm = slicer.app.layoutManager()
    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

    # fette sulla lesione: prima una trovata, se non ce ne sono una mancata
    if scelta is not None:
        segmentazione = scelta.GetSegmentation()
        for sid in ('Segment_1', 'Segment_2', 'Segment_3'):
            s = segmentazione.GetSegment(sid)
            if s is None:
                continue
            centro = scelta.GetSegmentCenterRAS(sid)
            if centro is not None:
                slicer.modules.markups.logic().JumpSlicesToLocation(centro[0], centro[1], centro[2], True)
                break

    vista = lm.threeDWidget(0).threeDView()
    vista.resetFocalPoint()
    vista.resetCamera()
    print(f'Aperto {os.path.basename(cartella)}, ricetta {ricetta}. Volumi: T1, FLAIR. '
          f'Segmentazioni: {", ".join("confronto " + k for k in RICETTE)} (visibile: {ricetta}), cervello.')
    return t1, fl, scelta


if __name__ == '__main__':
    # avviato con: Slicer --python-script apri_in_slicer.py FSM_046 [Z|A|D]
    # con exec() dalla console __name__ vale comunque '__main__' ma __file__ non esiste, e sys.argv
    # sono gli argomenti di Slicer: in quel caso non si apre nulla, si chiama apri() a mano
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = None
    argomenti = [a for a in sys.argv[1:] if not a.startswith('-')]
    if base and argomenti:
        paz = argomenti[0] if os.path.isabs(argomenti[0]) else os.path.join(base, argomenti[0])
        apri(paz, argomenti[1] if len(argomenti) > 1 else 'Z')
