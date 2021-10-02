# PCAM pre-trained models

## Folder structure

Unzip the pretrained models in this folder (`/path/to/pcam/data/trained_models/`), which should contain the following subfolders:

```
./3dmatch/
./3dmatch/soft_nbEnc_6_DistConf_4096/
./3dmatch/sparse_nbEnc_6_noBackprop_DistConf_4096/
./kitti/
./kitti/soft_nbEnc_6_DistConf/
./kitti/sparse_nbEnc_6_noBackprop_DistConf/
./modelnet/                                             # Contains the models for modelnet unseen objects
./modelnet/soft_nbEnc_6_DistConf/
./modelnet/sparse_nbEnc_6_noBackprop_DistConf/
./modelnet_unseen/                                      # Contains the models for modelnet unseen categories
./modelnet_unseen/soft_nbEnc_6_DistConf/
./modelnet_unseen/sparse_nbEnc_6_noBackprop_DistConf/
./modelnet_noise/                                       # Contains the models for modelnet unseen objects with noise
./modelnet_noise/soft_nbEnc_6_DistConf/
./modelnet_noise/sparse_nbEnc_6_noBackprop_DistConf/
```
