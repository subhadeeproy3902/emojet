# EmoJet: print the layout of datasets whose format we can't read on the laptop (CPU only, ~1 min).
import pandas as pd
from huggingface_hub import hf_hub_download
for f in ["training_data_english.xlsx", "fine-tuning_data_hinglish.xlsx", "testing_data_hinglish.xlsx"]:
    p = hf_hub_download("BM2002/hinglish-sarcasm-dataset", f, repo_type="dataset")
    x = pd.read_excel(p)
    print(f"=== {f}: {x.shape}  columns={list(x.columns)}")
    for c in x.columns:
        if x[c].nunique() < 10:
            print(f"   {c}: {x[c].value_counts().to_dict()}")
    print(x.head(4).to_string()[:1200])
