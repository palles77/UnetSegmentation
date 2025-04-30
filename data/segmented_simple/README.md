# Info
Directory with segmentation results

# Analysis usage

```
Get-ChildItem -Path . -Directory | ForEach-Object { & "./analyze_segmentation.ps1" -directory $_.FullName }
```

# Best outcome

```
Directory Analysis Report
=========================
Directory Name: 8

Parameters:
-----------
Final Optimal Segmentation Window Step: 8

Processing Time Summary:
------------------------
Total Time Elapsed: 24,727.1100 seconds

Segmentation Quality Analysis (Jaccard Distance):
-------------------------------------------------
Total Samples: 150
Average Jaccard Distance: 0.5639
Standard Deviation: 0.0851
Minimum Jaccard Distance: 0.2642
Maximum Jaccard Distance: 0.7170

Interpretation:
---------------
- A lower Jaccard Distance means better segmentation accuracy.
- A high standard deviation suggests variability in segmentation performance.
- If the average Jaccard Distance is close to 0, the model performed well.
- If the maximum Jaccard Distance is high, some images were segmented poorly.
```

# Last Update
2025/03/31