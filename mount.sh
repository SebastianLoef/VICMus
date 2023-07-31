MSD_MOUNT=/dev/sdc
STANDARD_MOUNT=/dev/sdb

sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard $STANDARD_MOUNT

sudo mkdir -p /mnt/disks/msd
sudo mkdir -p /mnt/disks/standard

sudo mount -o discard,defaults $MSD_MOUNT /mnt/disks/msd
sudo mount -o discard,defaults $STANDARD_MOUNT /mnt/disks/standard

sudo mkdir -p /mnt/disks/standard/models
sudo chmod 777 /mnt/disks/standard/models
ln -s /mnt/disks/standard/models /home/sebastian_lof_epidemicsound_com/VICMus/data
ln -s /mnt/disks/msd/ /home/sebastian_lof_epidemicsound_com/VICMus/data/processed
