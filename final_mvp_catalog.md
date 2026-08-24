# Final MVP Candidate Review Catalog — External Generalization Fight (`data/fight2.mp4`)

**Video Specifications**: 1920x1080 @ 25.00 FPS | Total Frames: 5,254 (210.12s)
**Fighters**: Canelo Alvarez vs. Gennady Golovkin II
**Pipeline Status**: Frozen MVP Baseline (RobustScaleManager + 5-Frame Kinematic Coasting)
**Total Detected Strikes**: 168

| Event # | Timestamp | Frame | Attacker | Strike Type | Hand | Target | Pred. Outcome | Conf. | Dynamic Physical Diagnostics |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #001 | 0.12s | 3 | Fighter 1 | CROSS | Left | BODY | **MISSED** | 0.54 | `out-of-range feint (init=740.8px > 432.9px)` |
| #002 | 1.08s | 27 | Fighter 2 | CROSS | Right | BODY | **MISSED** | 0.43 | `out-of-range feint (init=634.5px > 409.5px)` |
| #003 | 3.36s | 84 | Fighter 1 | JAB | Left | BODY | **MISSED** | 0.56 | `clear miss (d_min=251.1px > 159.2px)` |
| #004 | 5.16s | 129 | Fighter 2 | CROSS | Left | HEAD | **MISSED** | 0.47 | `out-of-range feint (init=879.6px > 429.1px)` |
| #005 | 6.04s | 151 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.58 | `out-of-range feint (init=384.5px > 368.9px)` |
| #006 | 6.04s | 151 | Fighter 2 | CROSS | Right | UNKNOWN | **MISSED** | 0.48 | `out-of-range feint (init=1142.3px > 366.7px)` |
| #007 | 7.76s | 194 | Fighter 2 | JAB | Left | UNKNOWN | **BLOCKED** | 0.54 | `guard interception (d_min=36.7px, guard_thr=92.9px)` |
| #008 | 10.04s | 251 | Fighter 2 | HOOK | Left | BODY | **MISSED** | 0.26 | `out-of-range feint (init=711.9px > 506.9px)` |
| #009 | 10.20s | 255 | Fighter 2 | CROSS | Right | BODY | **MISSED** | 0.43 | `out-of-range feint (init=761.0px > 507.3px)` |
| #010 | 10.84s | 271 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.68 | `occluded/missing target` |
| #011 | 11.52s | 288 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.60 | `occluded/missing target` |
| #012 | 11.64s | 291 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.58 | `occluded/missing target` |
| #013 | 12.44s | 311 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.59 | `out-of-range feint (init=756.0px > 393.1px)` |
| #014 | 12.80s | 320 | Fighter 2 | CROSS | Right | HEAD | **MISSED** | 0.54 | `out-of-range feint (init=1236.8px > 393.1px)` |
| #015 | 13.20s | 330 | Fighter 1 | JAB | Left | BODY | **LANDED** | 0.66 | `clean contact (d_min=76.4px <= hit_thr=223.1px, SW_ref=278.8px)` |
| #016 | 13.20s | 330 | Fighter 1 | CROSS | Right | BODY | **LANDED** | 0.53 | `clean contact (d_min=146.2px <= hit_thr=223.1px, SW_ref=278.8px)` |
| #017 | 24.32s | 608 | Fighter 2 | JAB | Left | BODY | **MISSED** | 0.34 | `clear miss (d_min=136.2px > 124.5px)` |
| #018 | 24.52s | 613 | Fighter 2 | CROSS | Right | BODY | **MISSED** | 0.52 | `clear miss (d_min=294.6px > 122.0px)` |
| #019 | 25.56s | 639 | Fighter 2 | CROSS | Right | UNKNOWN | **LANDED** | 0.58 | `clean contact (d_min=29.6px <= hit_thr=106.1px, SW_ref=132.6px)` |
| #020 | 26.08s | 652 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.50 | `occluded/missing target` |
| #021 | 26.47s | 662 | Fighter 1 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.65 | `occluded/missing target` |
| #022 | 27.51s | 688 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.60 | `occluded/missing target` |
| #023 | 27.59s | 690 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.58 | `occluded/missing target` |
| #024 | 30.91s | 773 | Fighter 2 | HOOK | Right | UNKNOWN | **MISSED** | 0.42 | `clear miss (d_min=248.2px > 151.1px)` |
| #025 | 30.99s | 775 | Fighter 2 | HOOK | Left | UNKNOWN | **MISSED** | 0.46 | `clear miss (d_min=274.5px > 151.1px)` |
| #026 | 32.03s | 801 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.56 | `out-of-range feint (init=561.1px > 418.3px)` |
| #027 | 32.39s | 810 | Fighter 2 | CROSS | Right | UNKNOWN | **BLOCKED** | 0.62 | `guard interception (d_min=47.6px, guard_thr=100.2px)` |
| #028 | 32.83s | 821 | Fighter 1 | JAB | Right | BODY | **UNCERTAIN** | 0.67 | `ambiguous boundary zone (115.3px < d_min=17.8px <= 129.7px)` |
| #029 | 33.23s | 831 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.58 | `occluded/missing target` |
| #030 | 34.55s | 864 | Fighter 2 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.45 | `occluded/missing target` |
| #031 | 35.83s | 896 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.67 | `occluded/missing target` |
| #032 | 37.83s | 946 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.53 | `occluded/missing target` |
| #033 | 38.31s | 958 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.70 | `occluded/missing target` |
| #034 | 38.83s | 971 | Fighter 2 | JAB | Left | UNKNOWN | **LANDED** | 0.56 | `clean contact (d_min=122.8px <= hit_thr=132.8px, SW_ref=166.0px)` |
| #035 | 45.67s | 1142 | Fighter 2 | JAB | Right | HEAD | **UNCERTAIN** | 0.45 | `ambiguous boundary zone (125.2px < d_min=127.7px <= 140.9px)` |
| #036 | 45.91s | 1148 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.39 | `occluded/missing target` |
| #037 | 47.03s | 1176 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.40 | `occluded/missing target` |
| #038 | 47.99s | 1200 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.67 | `occluded/missing target` |
| #039 | 49.47s | 1237 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.57 | `occluded/missing target` |
| #040 | 50.35s | 1259 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.50 | `occluded/missing target` |
| #041 | 50.51s | 1263 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.50 | `occluded/missing target` |
| #042 | 53.15s | 1329 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.42 | `occluded/missing target` |
| #043 | 54.15s | 1354 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.48 | `occluded/missing target` |
| #044 | 55.95s | 1399 | Fighter 2 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.34 | `occluded/missing target` |
| #045 | 56.47s | 1412 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.45 | `occluded/missing target` |
| #046 | 57.23s | 1431 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.69 | `occluded/missing target` |
| #047 | 57.55s | 1439 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.73 | `occluded/missing target` |
| #048 | 58.03s | 1451 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.38 | `occluded/missing target` |
| #049 | 59.15s | 1479 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.58 | `occluded/missing target` |
| #050 | 62.59s | 1565 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.70 | `occluded/missing target` |
| #051 | 63.87s | 1597 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.54 | `occluded/missing target` |
| #052 | 64.63s | 1616 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.56 | `occluded/missing target` |
| #053 | 66.59s | 1665 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.44 | `occluded/missing target` |
| #054 | 66.67s | 1667 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.63 | `occluded/missing target` |
| #055 | 67.51s | 1688 | Fighter 2 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.51 | `occluded/missing target` |
| #056 | 67.59s | 1690 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.39 | `occluded/missing target` |
| #057 | 68.95s | 1724 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.70 | `occluded/missing target` |
| #058 | 68.99s | 1725 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.63 | `occluded/missing target` |
| #059 | 70.91s | 1773 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.43 | `occluded/missing target` |
| #060 | 71.11s | 1778 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.60 | `occluded/missing target` |
| #061 | 72.43s | 1811 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.48 | `occluded/missing target` |
| #062 | 74.91s | 1873 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.51 | `occluded/missing target` |
| #063 | 75.03s | 1876 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.75 | `occluded/missing target` |
| #064 | 80.58s | 2015 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.45 | `occluded/missing target` |
| #065 | 81.50s | 2038 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.58 | `occluded/missing target` |
| #066 | 83.22s | 2081 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.58 | `occluded/missing target` |
| #067 | 84.18s | 2105 | Fighter 2 | JAB | Left | BODY | **UNCERTAIN** | 0.70 | `ambiguous boundary zone (125.0px < d_min=118.6px <= 140.7px)` |
| #068 | 84.22s | 2106 | Fighter 2 | HOOK | Right | BODY | **MISSED** | 0.61 | `clear miss (d_min=143.9px > 140.7px)` |
| #069 | 93.78s | 2345 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.50 | `occluded/missing target` |
| #070 | 98.06s | 2452 | Fighter 2 | JAB | Right | BODY | **MISSED** | 0.53 | `out-of-range feint (init=334.1px > 297.8px)` |
| #071 | 100.22s | 2506 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.59 | `occluded/missing target` |
| #072 | 102.78s | 2570 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.59 | `clear miss (d_min=220.0px > 107.9px)` |
| #073 | 105.46s | 2637 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.63 | `occluded/missing target` |
| #074 | 105.70s | 2643 | Fighter 2 | UPPERCUT | Right | UNKNOWN | **UNCERTAIN** | 0.47 | `occluded/missing target` |
| #075 | 106.50s | 2663 | Fighter 2 | CROSS | Right | UNKNOWN | **MISSED** | 0.65 | `out-of-range feint (init=339.8px > 299.7px)` |
| #076 | 107.30s | 2683 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.70 | `occluded/missing target` |
| #077 | 108.66s | 2717 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.62 | `out-of-range feint (init=601.0px > 352.9px)` |
| #078 | 109.34s | 2734 | Fighter 2 | CROSS | Right | UNKNOWN | **LANDED** | 0.59 | `clean contact (d_min=98.9px <= hit_thr=150.2px, SW_ref=187.7px)` |
| #079 | 109.46s | 2737 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.68 | `out-of-range feint (init=557.3px > 454.5px)` |
| #080 | 111.58s | 2790 | Fighter 2 | JAB | Left | BODY | **MISSED** | 0.62 | `clear miss (d_min=190.6px > 134.4px)` |
| #081 | 111.82s | 2796 | Fighter 2 | UPPERCUT | Right | BODY | **LANDED** | 0.27 | `clean contact (d_min=44.7px <= hit_thr=119.4px, SW_ref=149.3px)` |
| #082 | 116.34s | 2909 | Fighter 2 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.41 | `occluded/missing target` |
| #083 | 116.54s | 2914 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.58 | `occluded/missing target` |
| #084 | 117.50s | 2938 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.37 | `occluded/missing target` |
| #085 | 117.70s | 2943 | Fighter 2 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.48 | `occluded/missing target` |
| #086 | 121.18s | 3030 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.70 | `occluded/missing target` |
| #087 | 122.26s | 3057 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.76 | `occluded/missing target` |
| #088 | 124.58s | 3115 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.51 | `occluded/missing target` |
| #089 | 125.38s | 3135 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.43 | `occluded/missing target` |
| #090 | 126.10s | 3153 | Fighter 2 | CROSS | Right | UNKNOWN | **MISSED** | 0.68 | `clear miss (d_min=336.9px > 122.9px)` |
| #091 | 130.74s | 3269 | Fighter 2 | JAB | Left | UNKNOWN | **BLOCKED** | 0.60 | `guard interception (d_min=12.8px, guard_thr=96.9px)` |
| #092 | 131.22s | 3281 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.59 | `occluded/missing target` |
| #093 | 132.01s | 3301 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.59 | `occluded/missing target` |
| #094 | 132.13s | 3304 | Fighter 1 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.37 | `occluded/missing target` |
| #095 | 132.93s | 3324 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.60 | `occluded/missing target` |
| #096 | 134.17s | 3355 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.60 | `occluded/missing target` |
| #097 | 134.29s | 3358 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.71 | `occluded/missing target` |
| #098 | 135.57s | 3390 | Fighter 1 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.51 | `occluded/missing target` |
| #099 | 137.13s | 3429 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.59 | `occluded/missing target` |
| #100 | 138.93s | 3474 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.61 | `clear miss (d_min=164.5px > 151.0px)` |
| #101 | 139.13s | 3479 | Fighter 2 | CROSS | Right | UNKNOWN | **MISSED** | 0.66 | `clear miss (d_min=250.3px > 151.0px)` |
| #102 | 143.21s | 3581 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.33 | `occluded/missing target` |
| #103 | 144.25s | 3607 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.68 | `occluded/missing target` |
| #104 | 145.17s | 3630 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.69 | `occluded/missing target` |
| #105 | 150.73s | 3769 | Fighter 2 | HOOK | Left | BODY | **MISSED** | 0.29 | `out-of-range feint (init=487.0px > 386.6px)` |
| #106 | 151.65s | 3792 | Fighter 1 | CROSS | Right | UNKNOWN | **MISSED** | 0.72 | `out-of-range feint (init=660.7px > 289.0px)` |
| #107 | 151.69s | 3793 | Fighter 1 | JAB | Left | UNKNOWN | **MISSED** | 0.58 | `out-of-range feint (init=676.5px > 289.0px)` |
| #108 | 155.09s | 3878 | Fighter 1 | JAB | Left | UNKNOWN | **MISSED** | 0.46 | `out-of-range feint (init=750.3px > 293.0px)` |
| #109 | 156.65s | 3917 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.59 | `occluded/missing target` |
| #110 | 157.21s | 3931 | Fighter 1 | HOOK | Left | UNKNOWN | **MISSED** | 0.54 | `out-of-range feint (init=1260.6px > 620.8px)` |
| #111 | 157.49s | 3938 | Fighter 1 | CROSS | Right | BODY | **MISSED** | 0.46 | `clear miss (d_min=406.2px > 223.0px)` |
| #112 | 158.05s | 3952 | Fighter 1 | HOOK | Left | BODY | **BLOCKED** | 0.33 | `guard interception (d_min=127.1px, guard_thr=147.7px)` |
| #113 | 158.49s | 3963 | Fighter 2 | CROSS | Left | HEAD | **MISSED** | 0.60 | `clear miss (d_min=182.4px > 135.5px)` |
| #114 | 158.73s | 3969 | Fighter 1 | CROSS | Right | BODY | **BLOCKED** | 0.59 | `guard interception (d_min=77.1px, guard_thr=90.4px)` |
| #115 | 158.85s | 3972 | Fighter 1 | CROSS | Left | BODY | **MISSED** | 0.62 | `clear miss (d_min=147.6px > 134.9px)` |
| #116 | 163.49s | 4088 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.46 | `occluded/missing target` |
| #117 | 164.89s | 4123 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.55 | `occluded/missing target` |
| #118 | 164.89s | 4123 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.65 | `occluded/missing target` |
| #119 | 165.77s | 4145 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.65 | `occluded/missing target` |
| #120 | 165.85s | 4147 | Fighter 1 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.55 | `occluded/missing target` |
| #121 | 167.45s | 4187 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.57 | `occluded/missing target` |
| #122 | 167.73s | 4194 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.54 | `occluded/missing target` |
| #123 | 168.41s | 4211 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.66 | `occluded/missing target` |
| #124 | 168.57s | 4215 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.71 | `occluded/missing target` |
| #125 | 169.37s | 4235 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.60 | `occluded/missing target` |
| #126 | 169.37s | 4235 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.61 | `occluded/missing target` |
| #127 | 170.25s | 4257 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.57 | `clear miss (d_min=324.3px > 144.5px)` |
| #128 | 170.45s | 4262 | Fighter 1 | JAB | Left | BODY | **MISSED** | 0.53 | `clear miss (d_min=264.5px > 158.1px)` |
| #129 | 170.49s | 4263 | Fighter 2 | JAB | Right | BODY | **UNCERTAIN** | 0.56 | `ambiguous boundary zone (126.8px < d_min=139.1px <= 142.7px)` |
| #130 | 172.17s | 4305 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.55 | `occluded/missing target` |
| #131 | 173.13s | 4329 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.61 | `occluded/missing target` |
| #132 | 173.45s | 4337 | Fighter 1 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.41 | `occluded/missing target` |
| #133 | 175.25s | 4382 | Fighter 1 | HOOK | Right | BODY | **BLOCKED** | 0.51 | `guard interception (d_min=70.1px, guard_thr=89.9px)` |
| #134 | 177.97s | 4450 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.65 | `occluded/missing target` |
| #135 | 179.37s | 4485 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.58 | `occluded/missing target` |
| #136 | 182.05s | 4552 | Fighter 2 | JAB | Left | BODY | **MISSED** | 0.68 | `clear miss (d_min=211.4px > 127.8px)` |
| #137 | 182.37s | 4560 | Fighter 2 | HOOK | Right | BODY | **MISSED** | 0.43 | `out-of-range feint (init=357.7px > 354.9px)` |
| #138 | 183.09s | 4578 | Fighter 2 | CROSS | Left | HEAD | **LANDED** | 0.48 | `clean contact (d_min=78.4px <= hit_thr=110.5px, SW_ref=138.2px)` |
| #139 | 183.85s | 4597 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.61 | `occluded/missing target` |
| #140 | 184.00s | 4601 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.42 | `occluded/missing target` |
| #141 | 184.64s | 4617 | Fighter 1 | JAB | Left | UNKNOWN | **MISSED** | 0.59 | `out-of-range feint (init=327.7px > 297.6px)` |
| #142 | 184.96s | 4625 | Fighter 2 | CROSS | Right | BODY | **UNCERTAIN** | 0.47 | `ambiguous boundary zone (130.7px < d_min=133.6px <= 147.0px)` |
| #143 | 185.36s | 4635 | Fighter 2 | UPPERCUT | Left | UNKNOWN | **UNCERTAIN** | 0.73 | `occluded/missing target` |
| #144 | 187.04s | 4677 | Fighter 1 | JAB | Right | HEAD | **MISSED** | 0.39 | `clear miss (d_min=204.5px > 129.9px)` |
| #145 | 187.96s | 4700 | Fighter 1 | CROSS | Left | HEAD | **MISSED** | 0.56 | `clear miss (d_min=178.7px > 129.9px)` |
| #146 | 188.28s | 4708 | Fighter 1 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.61 | `occluded/missing target` |
| #147 | 188.88s | 4723 | Fighter 1 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.57 | `occluded/missing target` |
| #148 | 189.68s | 4743 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.64 | `occluded/missing target` |
| #149 | 190.36s | 4760 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.45 | `occluded/missing target` |
| #150 | 190.60s | 4766 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.59 | `occluded/missing target` |
| #151 | 191.16s | 4780 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.62 | `occluded/missing target` |
| #152 | 191.52s | 4789 | Fighter 1 | JAB | Left | UNKNOWN | **LANDED** | 0.63 | `clean contact (d_min=58.4px <= hit_thr=108.7px, SW_ref=135.9px)` |
| #153 | 194.20s | 4856 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.55 | `occluded/missing target` |
| #154 | 194.92s | 4874 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.65 | `occluded/missing target` |
| #155 | 196.16s | 4905 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.57 | `occluded/missing target` |
| #156 | 197.56s | 4940 | Fighter 2 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.43 | `occluded/missing target` |
| #157 | 197.76s | 4945 | Fighter 2 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.73 | `occluded/missing target` |
| #158 | 199.80s | 4996 | Fighter 2 | JAB | Left | UNKNOWN | **MISSED** | 0.62 | `out-of-range feint (init=458.1px > 361.1px)` |
| #159 | 200.32s | 5009 | Fighter 2 | HOOK | Right | UNKNOWN | **UNCERTAIN** | 0.49 | `occluded/missing target` |
| #160 | 203.76s | 5095 | Fighter 1 | CROSS | Right | UNKNOWN | **MISSED** | 0.59 | `out-of-range feint (init=457.5px > 327.4px)` |
| #161 | 204.28s | 5108 | Fighter 1 | JAB | Left | UNKNOWN | **MISSED** | 0.56 | `out-of-range feint (init=593.0px > 340.8px)` |
| #162 | 205.08s | 5128 | Fighter 1 | JAB | Left | UNKNOWN | **MISSED** | 0.62 | `out-of-range feint (init=595.2px > 339.6px)` |
| #163 | 205.92s | 5149 | Fighter 1 | JAB | Left | UNKNOWN | **UNCERTAIN** | 0.54 | `occluded/missing target` |
| #164 | 206.44s | 5162 | Fighter 1 | CROSS | Right | UNKNOWN | **MISSED** | 0.46 | `out-of-range feint (init=493.9px > 418.1px)` |
| #165 | 207.36s | 5185 | Fighter 1 | JAB | Left | UNKNOWN | **MISSED** | 0.63 | `out-of-range feint (init=548.7px > 357.2px)` |
| #166 | 207.40s | 5186 | Fighter 1 | CROSS | Right | UNKNOWN | **MISSED** | 0.69 | `out-of-range feint (init=371.1px > 357.2px)` |
| #167 | 208.32s | 5209 | Fighter 1 | HOOK | Left | UNKNOWN | **UNCERTAIN** | 0.48 | `occluded/missing target` |
| #168 | 208.64s | 5217 | Fighter 1 | CROSS | Right | UNKNOWN | **UNCERTAIN** | 0.60 | `occluded/missing target` |