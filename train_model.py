from kaggle.api.kaggle_api_extended import KaggleApi
import os, pandas as pd

HERE = os.path.dirname(__file__)
csv_path = os.path.join(HERE, "news.csv")

if not os.path.exists(csv_path):
    print("📥 Downloading dataset from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("clmentbisaillon/fake-and-real-news-dataset", path=HERE, unzip=True)

    # Merge Fake + True into one CSV
    df_true = pd.read_csv(os.path.join(HERE, "True.csv"))
    df_fake = pd.read_csv(os.path.join(HERE, "Fake.csv"))
    df_true["label"] = "real"
    df_fake["label"] = "fake"
    df = pd.concat([df_true[["text","label"]], df_fake[["text","label"]]])
    df.to_csv(csv_path, index=False)
    print("✅ Dataset prepared:", csv_path)
