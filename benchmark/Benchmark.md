# Standard Benchmark

NGC6888 data is processed with the following parameters:

```
basepath=Path('data/astro')
source_dirs=['NGC6888']
pattern='*.tiff'
patchsize=128

raw_data = RawData.from_folder_n2n(basepath, source_dirs=source_dirs, axes='YXC', pattern=pattern, preprocessor=STFPreProcessor(C=-2.8,B=0.25), imageloader=None)

create_patches_hdf5(
    raw_data, 
    patch_size=(patchsize,patchsize,3),
    normalization=None,
    n_patches_per_image=<num_patches>,
    save_file=save_file,
    patch_filter=None,
    overlap=False,
    shuffle=True,
    collapse_channel=True)
```

With <num_patches> set at 100 and 200.

The data processing step output for <num_patches> = 200 was a follows:

```
==================================================================
  190 raw image pairs x    1 transformations   =   190 images
  190 images     x 200 patches per image = 38000 patches in total
38000 patches    x    3 channel per patch = 114000 collapsed patches in total
==================================================================
Input data:
data\astro:, sources='['NGC6888']', axes='YXC', pattern='*.tiff'
==================================================================
Transformations:
1 x Identity
==================================================================
Patch size:
128 x 128 x 3
Collapsed Patch size:
128 x 128 x 1
==================================================================
100%|██████████| 190/190 [05:08<00:00,  1.62s/it  Last Image Shape (1722, 2179, 3)]
Shuffling HDF5 data via copy. This can take time for large datasets...
Destination Shape (114000, 128, 128, 1)
Destination Chunks (256, 128, 128, 1)
100%|██████████| 114000/114000 [01:39<00:00, 1144.89it/s]
Saved as hdf5 data to data\astro\NGC6888_ppi200_p128_STFPreProcessor_NoNorm_ChanCol.hdf5.
```

This output can be used as a benchmark for processing of your data on your system.

## Training

The two sets of data for 100 ppi & 200 ppi were trained using the following config:

```
probabilistic=<is_prob>
train_loss = <loss>
train_batch_size = 128
train_epochs=50
train_passes=4
train_steps_per_epoch = int(np.ceil(((len(train_data)*train_passes)/train_epochs)/train_batch_size))

config = Config('SYXC', n_channel_in=1, n_channel_out=1, unet_kern_size=3,
    train_loss = train_loss,
    probabilistic=probabilistic, 
    train_steps_per_epoch=train_steps_per_epoch, 
    train_epochs=train_epochs, 
    train_batch_size=train_batch_size)

```

With:

* <is_prob> = True and`<loss>` = 'laplace' and,
* <is_prob> = False and`<loss>` = 'mse' and,

## Benchmark Results

Testing of the 4 models against the `CrescentNebula-NoSt-Deep.tiff` image with the following config for prediction:

```
...
  
  model.predict(x[:,:,i],testaxes, normalizer=STFNormalizer(C=-2.8,B=0.25), resizer=PadAndCropResizer(), n_tiles = (2, 4))

...
```

Provided bellow are the result outputs available [here](https://1drv.ms/u/s!AvWEkn9Anb_Nq9Aw52Xs3LuYEcq_rg?e=EexXxL) for comparison.

| Samples Per Image | Model/Loss          | Output                                                                                                                    |
| ----------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 100               | Standard/mse        | `NGC6888_ppi100_p128_STFPreProcessor_NoNorm_ChanCol_Lmse-PRB-False_B128_SPE33_E50_RGB_CrescentNebula-NoSt-Deep.fits`    |
| 100               | Probalistic/laplase | `NGC6888_ppi100_p128_STFPreProcessor_NoNorm_ChanCol_Llaplace-PRB-True_B128_SPE33_E50_RGB_CrescentNebula-NoSt-Deep.fits` |
| 200               | Standard/mse        | `NGC6888_ppi200_p128_STFPreProcessor_NoNorm_ChanCol_Lmse-PRB-False_B128_SPE65_E50_RGB_CrescentNebula-NoSt-Deep.fits`    |
| 200               | Probalistic/laplase | `NGC6888_ppi200_p128_STFPreProcessor_NoNorm_ChanCol_Llaplace-PRB-True_B128_SPE65_E50_RGB_CrescentNebula-NoSt-Deep.fits` |

## Benchmark Results by Normalization


The following is a comparison on learning outcome using different source image and patch normalization.

| Target - Samples/Patch - Channels                        | Normalize                                   | Model/Loss          | Train Params      | History                                                                                                            | Result                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------- | ------------------------------------------- | ------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled          | NoPreProcessor<br />norm_simple             | Standard/mse        | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_simple_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1.png)           | ![NGC6888_ppi150_p128_NoPreProcessor__norm_simple_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_simple_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)                     |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled          | NoPreProcessor<br />norm_percentiles        | Standard/mse        | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_percentiles_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1.png)      | ![NGC6888_ppi150_p128_NoPreProcessor__norm_percentiles_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_percentiles_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)           |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled          | NoPreProcessor<br />norm_percentiles        | Probalistic/laplase | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_percentiles_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1.png)   | ![NGC6888_ppi150_p128_NoPreProcessor__norm_percentiles_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_percentiles_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)     |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled          | NoPreProcessor<br />norm_stf(C=-4,B=0.001)  | Standard/mse        | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.001_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1.png)     | ![NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.001_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.001_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)         |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled<br />xxx | NoPreProcessor<br />norm_stf(C=-4,B=0.001)  | Probalistic/laplase | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.001_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1.png)  | ![NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.001_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.001_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)   |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled<br />xxx | NoPreProcessor<br />norm_stf(C=-4,B=0.185)  | Standard/mse        | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.185_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1.png)     | ![NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.185_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.185_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)         |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled<br />xxx | NoPreProcessor<br />norm_stf(C=-4,B=0.185)  | Probalistic/laplase | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.185_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1.png)         | ![NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.185_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-4_0.185_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)                                                                                                                                                      |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled          | NoPreProcessor<br />norm_stf(C=-2.8,B=0.25) | Standard/mse        | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1.png)    | ![NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)       |
| NGC6888<br />150/128<br />ChanCol<br />Shuffled<br />xxx | NoPreProcessor<br />norm_stf(C=-2.8,B=0.25) | Probalistic/laplase | B128_SPE49_E50_V1 | ![](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1.png) | ![NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi150_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Llaplace-PRB-True_B128_SPE49_E50_V1_RGB_CrescentNebula-NoSt-Deep.png) |

## Benchmark Results by Loss

The following is a small comparison with a "custom" `hdr` loss function implemented additional to the csbdeep base framework.

The `hdr` loss function is simply a L1 type loss with `tanh` which attempts to de-emphasize the contribution of high intensity pixels to the learning model. 

As most of the shot or Poisson noise exists in low intensity pixels this may be beneficial.


| Target - Samples/Patch - Channels                        | Normalize                                   | Model/Loss          | Train Params      | History                                                                                                            | Result                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------- | ------------------------------------------- | ------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NGC6888<br />220/128<br />ChanCol<br />Shuffled          | NoPreProcessor<br />norm_stf(C=-2.8,B=0.25) | Standard/mse        | B128_SPE177_E50_V1 | ![](NGC6888_ppi220_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE177_E50_V1.png)    | ![NGC6888_ppi220_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE177_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi220_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lmse-PRB-False_B128_SPE177_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)       |
| NGC6888<br />220/128<br />ChanCol<br />Shuffled          | NoPreProcessor<br />norm_stf(C=-2.8,B=0.25) | hdr        | B128_SPE177_E50_V1 | ![](NGC6888_ppi220_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lhdr-PRB-False_B128_SPE177_E50_V1.png)    | ![NGC6888_ppi220_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lhdr-PRB-False_B128_SPE177_E50_V1_RGB_CrescentNebula-NoSt-Deep.fits](NGC6888_ppi220_p128_NoPreProcessor__norm_stf_-2.8_0.25_ChanCol_Shuffled_Lhdr-PRB-False_B128_SPE177_E50_V1_RGB_CrescentNebula-NoSt-Deep.png)       |