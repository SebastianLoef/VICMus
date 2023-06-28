import requests, zipfile, io, os
import json
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def download_lastfm_splits(save_path: str) -> None:
    print(f'Downloading Last.FM train, test splits to path {save_path} ...')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    zip_urls = ['http://millionsongdataset.com/sites/default/files/lastfm/lastfm_test.zip',
                'http://millionsongdataset.com/sites/default/files/lastfm/lastfm_train.zip']
    
    for url in zip_urls:
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(save_path)


def get_msd_tags(path: str) -> list:
    print("Extracting tags from Last.FM Train/Test splits ...")
    folders = ['lastfm_test', 'lastfm_train']
    tags = []

    for folder in [os.path.join(path, f) for f in folders]:
        for root, _, files in os.walk(folder):
            for file in files:
                path = os.path.join(root, file)

                with open(path) as f:
                    data = json.load(f)
                    if data['tags']:
                        tags.extend(data['tags'])

    return tags


def get_top_50_tags(tags: list) -> list:
    tag_set = [tag[0] for tag in tags]
    counted_tags = Counter(tag_set)

    top_50 = counted_tags.most_common(50)
    top_50_tags = [tup[0] for tup in top_50]

    return top_50_tags


def make_train_test_csvs(root: str, top_50_tags: list) -> None:
    
    print('Generating Last.FM train/test csv files...')
    root_path = root
    folders = ['lastfm_test', 'lastfm_train']

    for folder in [os.path.join(root, f) for f in folders]:
        tag_dict = {'paths': [],
                    'tags': [],
                    'top_50_tags': [],
                    'tagged_paths': []}
        name = folder.split('_')[-1]
        filename = f'msd_{name}_tags.csv'
        save_path = os.path.join(root_path, filename)

        if os.path.exists(save_path):
            continue

        for root, _, files in os.walk(folder):
            for file in files:
                path = os.path.join(root, file)

                with open(path) as f:
                    data = json.load(f)

                    if data['tags']:
                        song_tags = [tag[0] for tag in data['tags']]
                        tag_dict['paths'].append(path)

                        if not set(song_tags).isdisjoint(set(top_50_tags)):
                            tag_dict['tagged_paths'].append(path)
                            tag_dict['tags'].append(song_tags)
                            tag_dict['top_50_tags'].append(list(set(song_tags).intersection(set(top_50_tags))))

                        else:
                            tag_dict['tags'].append(np.NaN)
                            tag_dict['top_50_tags'].append(np.NaN)
                            tag_dict['tagged_paths'].append(np.NaN)

        df = pd.DataFrame(tag_dict)
        print(os.path.join(root_path, filename))
        df.to_csv(save_path)


def get_msd_train_test_dfs(root: str, msd_df_path: str):
    df_train = pd.read_csv(os.path.join(root, 'msd_train_tags.csv'), index_col=0)
    df_test = pd.read_csv(os.path.join(root, 'msd_test_tags.csv'), index_col=0)
    df_msd = pd.read_csv(msd_df_path, index_col=0)

    df_train['common_path'] = df_train['paths'].str.split(pat='/lastfm_(train|test)/', regex=True, expand=True)[2]
    df_train['common_path'] = df_train['common_path'].str.rsplit(pat='.', n=2, expand=True)[0]

    df_test['common_path'] = df_test['paths'].str.split(pat='/lastfm_(train|test)/', regex=True, expand=True)[2]
    df_test['common_path'] = df_test['common_path'].str.rsplit(pat='.', n=2, expand=True)[0]

    df_msd['audio_path'] = df_msd['audio_path'].str.split(pat='/', n=2, expand=True)[2]
    df_msd['common_path'] = df_msd['audio_path'].str.rsplit(pat='.', n=2, expand=True)[0]

    # Inner join on common_path
    df_train_msd = df_msd.merge(df_train)
    df_test_msd = df_msd.merge(df_test)

    return df_train_msd, df_test_msd


def add_binaries(df: pd.DataFrame, top_50_tags: list):
    mlb = MultiLabelBinarizer()
    mlb.fit([top_50_tags])
    df['binaries'] = df.apply(lambda x: mlb.transform([eval(x['top_50_tags'])])[0], axis=1)

    return df


def save_df_to_csv(save_path: str, df: pd.DataFrame, columns: list=['audio_path', 'binaries', 'top_50_tags']):

    df = df[df['can_load'] == True]
    df[columns].to_csv(save_path, index=False)


def get_data_subset(df: pd.DataFrame, 
                    sample_rates: list=[22050, 44100], 
                    song_lengths: list=[30, 60]):
    
    # Extract song length in seconds
    df['song_s'] = df['song_len'] / df['song_sr']
    index_list = []

    for length in song_lengths:
        # Allow +- 1s deviation for approximate lengths
        index_list.extend(df[df['song_s'].between(length-1, length+1)].index)

    return df[(df['song_sr'].isin(sample_rates)) &
              (df.index.isin(index_list))]

def save_binaries_and_paths_to_npy(df: pd.DataFrame, path: str, split:str):
    print(f'saving binaries npy format to {path}')
    np.save(file=os.path.join(path, split + '_binaries.npy'), arr=df['binaries'].values)
    np.save(file=os.path.join(path, split + '_files.npy'), arr=df['audio_path'].values)
    

def get_full_msd_files_npy(path: str, msd_df_path: str, min_audio_len: int=29, sample_rates: list=[22050, 44100]):

    msd_df = pd.read_csv(msd_df_path, index_col=0)

    msd_df['song_s'] = msd_df['song_len'] / msd_df['song_sr']
    msd_df['path'] = msd_df['audio_path'].str.split('audio/', expand=True)[1]
    npy = msd_df[(msd_df['song_s'] > min_audio_len) &
                 (msd_df['song_sr'].isin(sample_rates)) &
                 (msd_df['can_load'] == True)].path.values
    
    np.save(file=os.path.join(path, 'msd_file_list.npy'), arr=npy)

def main(MSD_path, msd_df_path):
    # Download msd splits
    download_lastfm_splits(save_path=MSD_path)

    # Get msd tags and extract top 50
    tags = get_msd_tags(path=MSD_path)
    top_50_tags = get_top_50_tags(tags)

    # Create and save train/test split dataframes
    make_train_test_csvs(root=MSD_path, top_50_tags=top_50_tags)

    # get msd train test evailability 
    train, test = get_msd_train_test_dfs(root=MSD_path, msd_df_path=msd_df_path)

    # Drop NaN values 
    train = train[train['top_50_tags'].notna()]
    test = test[test['top_50_tags'].notna()]

    # Add binaries
    train_binaries = add_binaries(train, top_50_tags)
    test_binaries = add_binaries(test, top_50_tags)

    # Reduce to wanted sample rate and song length
    df_train = get_data_subset(df=train_binaries,
                               sample_rates=[22050, 44100],
                               song_lengths=[30, 60])
        
    df_test = get_data_subset(df=test_binaries,
                              sample_rates=[22050, 44100],
                              song_lengths=[30, 60])

    # Get train / val split from df_test
    df_val, df_test = train_test_split(df_test, train_size=0.7)

    models = [(df_train, 'train'),
              (df_test, 'test'),
              (df_val, 'valid')]

    for data, split in models:

        save_df_to_csv(save_path=os.path.join(MSD_path, split + '.csv'),
                       df=data,
                       columns=['audio_path', 'binaries', 'top_50_tags', 'song_len', 'song_sr'])
        
        save_binaries_and_paths_to_npy(df=data, 
                                       path=MSD_path,
                                       split=split)

    get_full_msd_files_npy(path=MSD_path,
                           msd_df_path=msd_df_path,
                           min_audio_len=29, 
                           sample_rates=[22050, 44100])

if __name__ == '__main__':
    main(MSD_path='data/processed/MSD', msd_df_path='./msd_df.csv')

