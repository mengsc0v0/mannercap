# Command -> realised response (continuity audit)

| quantity | label | grades | Spearman (all) | monotone command range | Spearman (range) | R2 (range) | slope | realised range in monotone range | reachable realised (all) | invalid/total | repeatability | tie fraction | failed criteria |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wipe/sweep speed | **continuous** | 8 | 0.943 | 0.03..0.2 (9 lv) | 0.988 | 0.97 | 0.91 | 0.0451..0.19 m/s | 0.0323..0.266 | 5/84 | 0.18 | 0 | - |
| wipe/press force | **continuous** | 7 | 0.943 | 10..45 (8 lv) | 0.986 | 0.976 | 0.991 | 9.45..44.1 N | 9.29..72.1 | 9/72 | 0.0858 | 0 | - |
| lift/grip hold force | **graded(7)** | 7 | 0.978 | 4..34 (16 lv) | 0.974 | 0.955 | 1.16 | 2.1..40 N | 1.44..40 | 0/126 | 0.152 | 0.267 | staircase |
| lift/approach speed | **continuous** | 14 | 0.998 | 0.02..0.6 (14 lv) | 0.998 | 0.988 | 0.243 | 0.00398..0.148 m/s | 0.00356..0.152 | 0/84 | 0.057 | 0 | - |
| lift/descend speed | **continuous** | 14 | 0.998 | 0.02..0.6 (14 lv) | 0.998 | 1 | 0.268 | 0.00475..0.159 m/s | 0.0045..0.163 | 0/84 | 0.0141 | 0 | - |
| lift/lift speed | **continuous** | 14 | 0.998 | 0.02..0.6 (14 lv) | 0.998 | 1 | 0.26 | 0.00304..0.154 m/s | 0.00278..0.154 | 0/84 | 0.0151 | 0 | - |
| stack/grip hold force | **graded(6)** | 6 | 0.971 | 4..36 (9 lv) | 0.965 | 0.966 | 1.15 | 0.939..39.6 N | 0.937..39.6 | 0/60 | 2.66e-05 | 0.25 | staircase |
| stack/approach speed | **continuous** | 12 | 0.996 | 0.03..0.6 (13 lv) | 0.996 | 0.994 | 0.243 | 0.00584..0.149 m/s | 0.00476..0.152 | 7/84 | 0.108 | 0 | - |
| stack/descend speed | **continuous** | 13 | 0.997 | 0.03..0.6 (13 lv) | 0.997 | 0.993 | 0.257 | 0.00711..0.152 m/s | 0.00667..0.158 | 7/84 | 0.0405 | 0 | - |
| stack/lift speed | **continuous** | 13 | 0.997 | 0.03..0.6 (13 lv) | 0.997 | 1 | 0.257 | 0.0054..0.152 m/s | 0.00499..0.155 | 7/84 | 0.0268 | 0 | - |
| stack/transport speed | **continuous** | 13 | 0.996 | 0.03..0.6 (13 lv) | 0.996 | 0.999 | 0.265 | 0.00394..0.154 m/s | 0.00367..0.162 | 7/84 | 0.0935 | 0 | - |
| stack/place speed | **continuous** | 13 | 0.997 | 0.03..0.6 (13 lv) | 0.997 | 0.985 | 0.243 | 0.0091..0.143 m/s | 0.00104..0.15 | 7/84 | 0.0188 | 0 | - |
| stack/retreat speed | **continuous** | 13 | 0.997 | 0.03..0.6 (13 lv) | 0.997 | 1 | 0.26 | 0.00754..0.156 m/s | 0.0074..0.156 | 7/84 | 0.0117 | 0 | - |

Labels: `continuous` = over the monotone range Spearman>=0.95, R2>=0.90 and >=5 separable grades (AUC>=0.9); `graded(k)` = monotone but only k separable grades; `not_controllable` = no monotone range. Invalid = task failed, no valid measurement, or the command level succeeded in < 50 % of episodes (unusable level). repeatability = median over levels of (p90-p10)/median across seeds.
