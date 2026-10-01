# Componenti da togliere dalle annotazioni AInCP

Prodotto da `scripts/elenco_residui_annotazione.py`, **non modificare a mano**.

Componenti fino a 1 mm³ (decisione del 23 settembre: residui del tracciato), più quelle fuori dal cervello di qualunque dimensione. Spazio: la FLAIR originale di ciascun paziente. Indici di voxel contati da 1 (MRIcron); coordinate in mm dello scanner dall'header. L'indice *i* è la fetta sagittale.

| paziente | file ROI | fenotipo | voxel | mm³ | fetta sagittale *i* | *j* | *k* | x mm | y mm | z mm | nota |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSM_028 | `ROI_2.nii` | cistica | 1 | 1.0 | 105 | 131 | 129 | -26.0 | 11.5 | -11.2 |  |
| FSM_028 | `ROI_1.nii` | gliosi | 1 | 1.0 | 118 | 78 | 131 | -36.9 | -42.0 | -9.7 |  |
| FSM_028 | `ROI_2.nii` | cistica | 1 | 1.0 | 118 | 128 | 76 | -36.6 | 8.1 | -64.7 | **fuori dal cervello** |
| FSM_028 | `ROI_2.nii` | cistica | 1 | 1.0 | 118 | 134 | 86 | -37.2 | 14.0 | -54.7 | **fuori dal cervello** |
| FSM_028 | `ROI_2.nii` | cistica | 1 | 1.0 | 119 | 118 | 226 | -43.6 | -2.2 | 85.1 | **fuori dal cervello** |
| FSM_030 | `ROI_1.nii` | gliosi | 2 | 2.0 | 91 | 45 | 127 | -8.8 | -77.7 | -8.9 | **fuori dal cervello** |
| FSM_032 | `ROI_1.nii` | gliosi | 1 | 0.2 | 119 | 197 | 269 | -22.6 | -28.4 | 39.4 |  |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 122 | 191 | 378 | -26.9 | -31.3 | 93.8 |  |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 135 | 223 | 384 | -37.1 | -15.1 | 96.4 |  |
| FSM_032 | `ROI_2.nii` | cistica | 2 | 0.4 | 135 | 283 | 241 | -34.0 | 15.1 | 24.9 |  |
| FSM_032 | `ROI_1.nii` | gliosi | 1 | 0.2 | 150 | 143 | 277 | -48.1 | -54.9 | 42.5 |  |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 150 | 145 | 319 | -48.8 | -53.9 | 63.5 |  |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 189 | 247 | 260 | -77.8 | -2.3 | 32.9 | **fuori dal cervello** |
| FSM_032 | `ROI_2.nii` | cistica | 1 | 0.2 | 190 | 207 | 275 | -79.3 | -22.2 | 40.4 | **fuori dal cervello** |
| FSM_036 | `ROI_1.nii` | gliosi | 1 | 1.0 | 56 | 114 | 173 | 23.7 | -0.9 | 59.1 |  |
| FSM_036 | `ROI_1.nii` | gliosi | 1 | 1.0 | 71 | 113 | 173 | 8.7 | -1.9 | 59.5 |  |
| FSM_036 | `ROI_4.nii` | encefalomalacia | 1 | 1.0 | 122 | 115 | 159 | -42.6 | 0.1 | 46.9 |  |
| FSM_036 | `ROI_1.nii` | gliosi | 1 | 1.0 | 142 | 129 | 190 | -61.7 | 14.1 | 78.5 | **fuori dal cervello** |
| FSM_040 | `ROI_2.nii` | cistica | 1 | 0.2 | 37 | 175 | 308 | -59.0 | -31.8 | 26.3 |  |
| FSM_040 | `ROI_4.nii` | encefalomalacia | 1 | 0.2 | 51 | 230 | 193 | -42.5 | -5.5 | -30.6 | **fuori dal cervello** |
| FSM_041 | `ROI_1.nii` | gliosi | 4 | 0.8 | 141 | 259 | 306 | 31.6 | -0.2 | 38.4 |  |
| FSM_045 | `ROI_1.nii` | gliosi | 1 | 0.2 | 61 | 401 | 394 | -23.4 | 73.5 | 89.3 | **fuori dal cervello** |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 119 | 167 | 276 | 7.9 | 0.9 | 40.8 |  |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 124 | 137 | 164 | 6.2 | -28.0 | -22.5 |  |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 137 | 231 | 207 | -0.6 | 32.1 | -6.8 |  |
| FSM_049 | `ROI_1.nii` | gliosi | 3 | 0.65 | 139 | 203 | 270 | -3.4 | 21.9 | 33.3 |  |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 141 | 205 | 264 | -4.4 | 22.5 | 29.5 |  |
| FSM_049 | `ROI_2.nii` | cistica | 3 | 0.65 | 143 | 166 | 274 | -6.5 | 0.5 | 39.3 |  |
| FSM_049 | `ROI_1.nii` | gliosi | 4 | 0.86 | 147 | 245 | 203 | -6.3 | 39.9 | -10.4 |  |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 152 | 202 | 289 | -11.5 | 23.4 | 44.3 |  |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 155 | 244 | 178 | -10.7 | 37.1 | -25.6 |  |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 172 | 132 | 205 | -23.3 | -26.1 | 1.3 |  |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 177 | 252 | 179 | -23.7 | 42.3 | -26.2 |  |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 179 | 232 | 282 | -27.0 | 40.9 | 36.7 | **fuori dal cervello** |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 206 | 223 | 187 | -41.8 | 26.4 | -19.1 | **fuori dal cervello** |
| FSM_049 | `ROI_2.nii` | cistica | 1 | 0.22 | 209 | 178 | 270 | -45.8 | 8.2 | 34.4 | **fuori dal cervello** |
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

Totale: **46** componenti, di cui 12 fuori dal cervello. Per paziente: FSM_028 5, FSM_030 1, FSM_032 8, FSM_036 4, FSM_040 2, FSM_041 1, FSM_045 1, FSM_049 14, FSM_054 8, FSM_056 2.
