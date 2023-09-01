from src.data.freemusicarchive import FreeMusicArchive
from src.data.gtzan import GTZAN
from src.data.magnatagatune import MagnaTagATune
from src.data.millionsongdataset import MillionSongDataset
from src.data.nsynth import NSynthInstrument, NSynthPitch

DATASETS = {
    "mtat": MagnaTagATune,
    "fma": FreeMusicArchive,
    "gtzan": GTZAN,
    "msd": MillionSongDataset,
    "nsynth_instrument": NSynthInstrument,
    "nsynth_pitch": NSynthPitch,
}
