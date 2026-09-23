# Componenti da togliere dalle annotazioni AInCP

Prodotto da `scripts/elenco_residui_annotazione.py`, **non modificare a mano**.

Componenti fino a 1 mm³ (decisione del 23 settembre: residui del tracciato), più quelle fuori dal cervello di qualunque dimensione. Spazio: la FLAIR originale di ciascun paziente. Indici di voxel contati da 1 (MRIcron); coordinate in mm dello scanner dall'header. L'indice *i* è la fetta sagittale.

| paziente | file ROI | fenotipo | voxel | mm³ | fetta sagittale *i* | *j* | *k* | x mm | y mm | z mm | nota |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSM_032 | `ROI_1.nii` | gliosi | 1 | 0.2 | 119 | 197 | 269 | -22.6 | -28.4 | 39.4 |  |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 122 | 191 | 378 | -26.9 | -31.3 | 93.8 |  |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 135 | 223 | 384 | -37.1 | -15.1 | 96.4 |  |
| FSM_032 | `ROI_2.nii` | cistica | 2 | 0.4 | 135 | 283 | 241 | -34.0 | 15.1 | 24.9 |  |
| FSM_032 | `ROI_1.nii` | gliosi | 1 | 0.2 | 150 | 143 | 277 | -48.1 | -54.9 | 42.5 |  |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 150 | 145 | 319 | -48.8 | -53.9 | 63.5 |  |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 189 | 247 | 260 | -77.8 | -2.3 | 32.9 | **fuori dal cervello** |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 190 | 207 | 275 | -79.3 | -22.2 | 40.4 | **fuori dal cervello** |
| FSM_040 | `ROI_2.nii` | cistica | 1 | 0.2 | 37 | 175 | 308 | -59.0 | -31.8 | 26.3 |  |
| FSM_040 | `ROI_4.nii` | encefalomalacia | 1 | 0.2 | 51 | 230 | 193 | -42.5 | -5.5 | -30.6 | **fuori dal cervello** |
| FSM_041 | `ROI_1.nii` | gliosi | 4 | 0.8 | 141 | 259 | 306 | 31.6 | -0.2 | 38.4 |  |
| FSM_045 | `ROI_1.nii` | gliosi | 1 | 0.2 | 61 | 401 | 394 | -23.4 | 73.5 | 89.3 | **fuori dal cervello** |
| FSM_054 | `ROI_1.nii` | gliosi | 4 | 0.88 | 92 | 238 | 321 | -23.7 | 7.8 | 35.9 |  |
| FSM_054 | `ROI_4.nii` | encefalomalacia | 2 | 0.44 | 100 | 193 | 378 | -32.6 | -13.1 | 62.4 |  |
| FSM_054 | `ROI_1.nii` | gliosi | 2 | 0.44 | 100 | 211 | 382 | -32.6 | -5.2 | 64.2 |  |
| FSM_054 | `ROI_2.nii` | cistica | 1 | 0.22 | 103 | 180 | 363 | -35.4 | -19.5 | 55.3 |  |
| FSM_054 | `ROI_1.nii` | gliosi | 2 | 0.44 | 104 | 213 | 380 | -36.6 | -4.2 | 63.2 |  |
| FSM_054 | `ROI_1.nii` | gliosi | 3 | 0.66 | 107 | 285 | 288 | -38.3 | 29.6 | 20.0 |  |
| FSM_054 | `ROI_1.nii` | gliosi | 3 | 0.66 | 107 | 312 | 305 | -38.8 | 42.4 | 28.0 |  |
| FSM_054 | `ROI_1.nii` | gliosi | 2 | 0.44 | 114 | 265 | 329 | -46.0 | 20.1 | 39.1 |  |
| FSM_056 | `ROI_2.nii` | cistica | 2 | 0.44 | 119 | 217 | 389 | -31.9 | -13.2 | 74.0 |  |
| FSM_056 | `ROI_1.nii` | gliosi | 1 | 0.22 | 150 | 253 | 318 | -61.7 | 7.5 | 42.3 |  |

Totale: **22** componenti, di cui 4 fuori dal cervello. Per paziente: FSM_032 8, FSM_040 2, FSM_041 1, FSM_045 1, FSM_054 8, FSM_056 2.
