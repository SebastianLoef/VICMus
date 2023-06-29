import os
import zipfile
import pandas as pd
from tqdm import tqdm
import opendatasets as od

import torchaudio


def compile_MagnaTagaTune():
    # check if the dataset is already compiled
    # if os.path.exists('data/processed/MagnaTagATune'):

    # if not, compile it
    # unzip mp3 files
    if not os.path.exists("data/processed/MagnaTagATune/"):
        os.system("unzip data/raw/mp3.zip -d data/processed/MagnaTagATune/")
        os.system(
            f"cp data/raw/clip_info_final.csv"
            f"data/raw/annotations_final.csv"
            f"data/processed/MagnaTagATune/"
        )
        for file in ["train", "test", "valid", "binary", "tags"]:
            os.system(f"cp data/raw/{file}.npy data/processed/MagnaTagATune/")

        for file in ["train_gt", "test_gt", "val_gt", "index"]:
            os.system(f"cp data/raw/{file}_mtt.tsv data/processed/MagnaTagATune/")

    # Extract the top 50 most common tags from annotations
    tags = pd.read_csv("data/raw/annotations_final.csv", sep="\t")
    # find top 50 tags

    top_50_tags = tags.loc[:, ~tags.columns.isin(["clip_id", "mp3_path"])]
    top_50_tags = (
        top_50_tags.sum(axis=0).sort_values(ascending=False).head(50).index.tolist()
    )
    tags = tags[["clip_id"] + top_50_tags]
    tags.to_csv("data/processed/MagnaTagATune/top_50_tags.csv", index=False)


def compile_GTZAN():
    if not os.path.exists("data/processed/gtzan/"):
        od.download(
            "https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification",
            data_dir="data/processed/",
        )
        os.system(
            "mv data/processed/gtzan-dataset-music-genre-classification/Data/ data/processed/gtzan/"
        )
        os.system("rm data/processed/gtzan-dataset-music-genre-classification/")

        os.system(
            """
        wget -O data/processed/gtzan/train_filtered.txt https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/train_filtered.txt
        wget -O data/processed/gtzan/valid_filtered.txt  https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/valid_filtered.txt 
        wget -O data/processed/gtzan/test_filtered.txt https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/test_filtered.txt
        """
        )


def main():
    compile_MagnaTagaTune()
    # compile_GTZAN()


if __name__ == "__main__":
    main()
