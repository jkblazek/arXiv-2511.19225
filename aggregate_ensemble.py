import glob
import pandas as pd

files = glob.glob("out/ensemble/results_trial_*.csv")
dfs = [pd.read_csv(f) for f in files]

df = pd.concat(dfs, keys=range(len(dfs)), names=["trial"])
df = df.reset_index(level=0).rename(columns={"level_0": "trial"})

# Average E_j across trials
agg = (
    df.groupby(["conn","j"])
      .agg(
          Ej_mean=("pavg","mean"),
          Ej_std=("pavg","std"),
          sqrtV_mean=("pstd","mean")
      )
      .reset_index()
)

agg.to_csv("out/results_mauct_ensemble.csv", index=False)
print("Wrote out/results_mauct_ensemble.csv")

