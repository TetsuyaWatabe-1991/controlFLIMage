# Obvious-death reject on the first pre

Reject a field when the first pre max projection shows a large round bleb and no continuous dendrite shaft. A bead sitting on a long shaft is kept. This rule is only for obvious death on the first view. It does not try to catch other-Z bleed, a drifted ROI, or a wrong uncaging plane.

The comparison figure is

`//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/obvious_dead_figure.png`

## Parameters

Edit these in `obvious_dead.py`. Lengths are micrometers. `xy_um` is read from the acquisition state and is not a free parameter:

`xy_um = 0.5 * (FOV_x / zoom / pixels_x + FOV_y / zoom / pixels_y)`

| Name | Value | Meaning |
|---|---|---|
| `BLEB_MIN` | 0.45 | Minimum bleb score. Dimensionless, after a 0–1 stretch. |
| `SHAFT_MAX_UM` | 4.25 | Maximum shaft length that still counts as “no shaft”. |
| `BLEB_SIGMA_UM` | 0.46, 0.68, 0.91, 1.21, 1.67 | LoG sigma for round blebs. |
| `RIDGE_SIGMA_UM` | 0.15, 0.30, 0.46 | Frangi sigma for the shaft. |
| `BLEB_RADIUS_FACTOR` | 1.8 | Disk erased around a bleb, in units of that sigma. |

These micrometers are the zoom-14 high-mag sampling the rule was tuned on (FOV 273 x 271 µm, 128 px, zoom 14, 0.152 µm/pixel). At that sampling, 4.25 µm is 28 pixels and the bleb sigmas are 3, 4.5, 6, 8, and 11 pixels. At zoom 15 (0.142 µm/pixel) the same 4.25 µm is 30 pixels.

`obviously_dead(image, xy_um, bleb_min=..., shaft_max_um=..., bleb_sigma_um=..., ridge_sigma_um=...)` accepts any of those lengths in micrometers.

## Steps

1. Use only the first pre. Max-project that frame over Z.
2. Stretch finite intensities from the 1st to the 99.5th percentile onto 0–1.
3. Bleb score. At each sigma in `BLEB_SIGMA_UM`, compute the scale-normalized Laplacian of Gaussian, `-sigma^2 * laplacian`. Bright round objects are positive. The bleb score is the strongest peak across those scales.
4. Remove blebs before measuring the shaft. On each scale, peaks above `max(0.5, 0.65 * that scale's maximum)` are blebs. Zero a disk of radius `1.8 * sigma` around each peak so the rim of a bleb is not counted as a dendrite.
5. Shaft. Run a Frangi ridge filter on what remains (`black_ridges=False`, sigmas in `RIDGE_SIGMA_UM`). Keep ridge pixels at or above 15% of the strongest ridge. Thin that mask to a one-pixel skeleton.
6. Shaft length is the longest 8-connected path in one skeleton piece, times `xy_um`. A diagonal step counts as one pixel, not 1.4 pixels. Shorter pieces are ignored. In the figure the measured piece is orange and the shorter pieces are blue.
7. Reject when `bleb >= 0.45` and `shaft <= 4.25 µm`. Both must be true. A bright bleb with a long remaining shaft is kept.

## What the figure shows

Three real first-pre fields. Columns, left to right: the max projection (5 µm scale bar), the bleb map with cyan circles on the disks that are erased, the image after those disks are removed, and the skeleton on the original image.

- Top row, `20260909_AP5_13_pos1__highmag_7_set0`. Bleb 0.60, shaft 2.1 µm. Reject. The round object is erased and nothing long remains.
- Middle row, `20260909_AP5_13_pos1__highmag_6_set1`. Bleb 0.52, shaft 7.3 µm. Keep. The blebs are removed and an orange shaft is still longer than 4.25 µm.
- Bottom row, `20260623_3_pos1__highmag_5_set2`. Bleb 0.53, shaft 2.5 µm. Reject, but this field was labeled keep. The dendrite is a string of beads, so the longest connected piece is short even though the eye can still follow the dendrite.

## Result on the 211 labeled first-pre fields

Cutoff `bleb >= 0.45` and `shaft <= 4.25 µm`.

| Group | Called dead |
|---|---|
| User-listed obvious death (10) | 10/10 |
| Other category 8, “might still be alive” (6) | 2/6 |
| Category 9 keep (195) | 4/195 |

Specificity on keeps is 191/195 = 97.9%. Keep shaft lengths are much longer than the cutoff (median 14.6 µm, 10th percentile 6.4 µm, 5th percentile 4.9 µm).

The two “might still be alive” calls:

- `20260909_cnt_4_pos1__highmag_7_set1`: bleb 0.53, shaft 3.34 µm
- `20260623_5_pos1__highmag_4_set2`: bleb 0.56, shaft 3.83 µm

The four keeps called dead:

- `20260623_3_pos1__highmag_5_set2`: bleb 0.53, shaft 2.55 µm (bottom row of the figure)
- `20260623_4_pos1__highmag_5_set1`: bleb 0.57, shaft 4.25 µm (sits on the cutoff)
- `20260909_cnt_4_pos1__highmag_4_set1`: bleb 0.46, shaft 3.19 µm
- `20260909_cnt_4_pos1__highmag_4_set2`: bleb 0.49, shaft 3.64 µm

Stating the sizes in micrometers moves the zoom-15 fields (20260623) relative to the old pixel cutoffs. At zoom 15 the same physical sigma covers about 7% more pixels, and 4.25 µm is 30 pixels rather than 28. That is why `20260623_5_pos1__highmag_4_set2` and `20260623_4_pos1__highmag_5_set1` are now called dead. Zoom-14 calls are unchanged.
