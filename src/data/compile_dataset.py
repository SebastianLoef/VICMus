import os
import zipfile

import opendatasets as od
import pandas as pd
import torchaudio
from tqdm import tqdm


def compile_MagnaTagaTune():
    # check if the dataset is already compiled
    # if os.path.exists('data/processed/MagnaTagATune'):

    # if not, compile it
    # unzip mp3 files
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


def compile_FreeMusicArchive():
    if not os.path.exists("data/processed/FMA/"):
        with zipfile.ZipFile("data/raw/fma_medium.zip", "r") as zip_ref:
            zip_ref.extractall("data/processed/FMA/")
    # add compile metadata
    if os.path.exists("data/processed/FMA/fma_medium/metadata.csv"):
        return 0
    print("Compiling metadata... for FMA")
    df = pd.read_csv(
        "data/processed/FMA/fma_medium/checksums",
        sep="  ",
        header=None,
        engine="python",
        names=["checksum", "mp3_path"],
    )
    df["length"] = 0
    broken_idxs = [
        316,
        977,
        10675,
        13146,
        15626,
        15627,
        15628,
        15629,
        15630,
        15631,
        15632,
        15633,
        15634,
        15836,
        16305,
        16959,
        20621,
        20780,
        21988,
        23620,
    ]
    for i in tqdm(range(df.shape[0])):
        if i in broken_idxs:
            continue
        path = os.path.join("data/processed/FMA/fma_medium", df.loc[i, "mp3_path"])
        df.loc[i, "length"] = torchaudio.info(path).num_frames

    df.to_csv("data/processed/FMA/fma_medium/metadata.csv", sep=",", index=False)


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
    compile_FreeMusicArchive()
    # compile_GTZAN()


if __name__ == "__main__":
    main()
