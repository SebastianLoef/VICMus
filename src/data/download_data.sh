# This is a script to download the data from the MagnaTagATune dataset
# if statemnt to check if the data is already downloaded

# a general function for downloading the data
# $1 is the url
# $2 is the name of the file
function download_data {
    if [ -f "data/raw/$2" ]; then
        echo "Data already downloaded"
    else
        echo "Downloading data"
        wget "$1" -O "data/raw/$2"
        echo "------------------------------"
    fi
}
# a function for splitting the last / from the url
# $1 is the url
function split_url {
    echo "$1" | rev | cut -d'/' -f1 | rev
}

data=(
    "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001"
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002"
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003"
    "https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv"
    "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/binary.npy"
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/tags.npy"
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/test.npy"
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/train.npy"
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/valid.npy"
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/train_gt_mtt.tsv"
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/val_gt_mtt.tsv"
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/test_gt_mtt.tsv"
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/index_mtt.tsv"
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
)
echo "################################"
echo "Downloading data"
echo "################################"
for i in ${data[@]};
do
    echo "Check if $i is already downloaded"
    download_data "$i" "$(split_url "$i")"
done

# concatenate the files
cat data/raw/mp3.zip.00* > data/raw/mp3.zip
echo "################################"
echo "Completed downloading data"
echo "################################"
