# Association Mining and Eye-Tracking Clustering

This repository demonstrates two complementary data mining techniques on website interaction data:

- association rule mining over scanpath-style page-area sequences,
- DBSCAN clustering over eye-tracking fixation coordinates.

The public version is fully reproducible and uses synthetic gaze/scanpath data with the same structure as a typical eye-tracking web usability study. No raw participant data or private experiment files are included.

## Why this matters

Eye-tracking data is naturally sequential and spatial. Association mining helps reveal which page areas are often visited together, while clustering helps identify dense regions of visual attention on the page.

In a real usability or behavioral analytics setting, this type of workflow can answer questions such as:

- Which page elements are frequently inspected together?
- Do users who look at the product hero also inspect price information or CTA buttons?
- Where do fixation points form stable attention clusters?
- Which fixations are spatial outliers or noise?

## Methods

The project implements:

- synthetic scanpath generation,
- frequent itemset mining,
- association rule metrics: support, confidence, and lift,
- synthetic fixation generation around page areas,
- DBSCAN clustering for spatial gaze patterns,
- diagnostic plots for item frequencies, association rules, and fixation clusters.

## Run

```bash
pip install -r requirements.txt
python association_mining_clustering.py
```

## Outputs

Running the script creates local CSV outputs under `outputs/` and diagnostic figures under `figures/`.

Included example figures:

- `figures/scanpath_area_frequency.png`
- `figures/association_rules_bubble.png`
- `figures/dbscan_fixation_clusters.png`

The generated CSV files are ignored by Git and can be recreated by rerunning the script.
