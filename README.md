# Lesion detection in paralisi cerebrale infantile

Progetto di tesi: **rilevamento automatico di lesioni cerebrali su MRI in bambini con paralisi
cerebrale**, con nnU-Net v2. Tre dataset pubblici (WMH, ATLAS 2.0, DBB) servono a costruire il metodo e i
pesi di partenza; il bersaglio finale è il dataset privato AInCP, di circa 10 pazienti.

**Inizia da [`docs/obiettivo-e-piano.md`](docs/obiettivo-e-piano.md)**, che spiega a cosa serve il
progetto e cosa va fatto adesso. L'indice completo della documentazione è in
[`docs/README.md`](docs/README.md), con lo stato di avanzamento.

I volumi originali restano in sola lettura su `/data01`; `data/raw/` contiene symlink, non copie.

```bash
source scripts/env.sh    # variabili nnU-Net, trainer del progetto, scelta GPU
```

Cartelle: `configs`, `data/raw`, `data/processed`, `notebooks`, `scripts`, `reports`, `docs`, `nnUNet`.
