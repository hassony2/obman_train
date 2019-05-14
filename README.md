# Get the code

`git clone https://github.com/hassony2/obman_train`
`cd obman_train`

# Download and prepare datasets

## Download the ObMan dataset

- Download the dataset following [these instructions](https://github.com/hassony2/obman)
- Create a [symlink](https://www.cyberciti.biz/faq/creating-soft-link-or-symbolic-link/) `ln -s path/to/downloaded/obman datasymlinks/obman`
- Download  ShapeNetCore v2 object meshes from the [ShapeNet official website](https://www.shapenet.org/)
- Create a symlink `ln -s /sequoia/data2/dataset/shapenet/ShapeNetCore.v2 datasymlinks/ShapeNetCore.v2`

Your data structure should now look like 

```
obman_train/
  datasymlinks/ShapeNetCore.v2
  datasymlinks/obman
```

## Download the First-Person Hand Action Benchmark dataset

- Follow the [official instructions](https://github.com/guiggh/hand_pose_action) to download the dataset


## Install the MANO PyTorch layer

- Follow the instructions from [here](https://github.com/hassony2/manopth)

### Download the MANO model files

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format mano_v*_*.zip). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the content of the *models* folder into the misc/mano folder
- Your structure should look like this:

```
obman_render/
  misc/
    mano/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
```


# Launch

`python traineval.py --atlas_predict_trans --atlas_predict_scale --atlas_mesh --mano_use_shape --mano_use_pca --freeze_batchnorm --atlas_separate_encoder`

# Acknowledgements

## AtlasNet code

Code related to [AtlasNet](http://imagine.enpc.fr/~groueixt/atlasnet/) is in large part adapted from the official [AtlasNet repository](https://github.com/ThibaultGROUEIX/AtlasNet). 
Thanks [Thibault](https://github.com/ThibaultGROUEIX/) for the provided code !

## Hand evaluation code

Code for computing hand evaluation metrics was reused from [hand3d](https://github.com/lmb-freiburg/hand3d), courtesy of [Christian Zimmermann](https://lmb.informatik.uni-freiburg.de/people/zimmermc/) with an easy-to-use interface!


## Laplacian regularization loss

[Code](https://github.com/akanazawa/cmr) for the laplacian regularization and precious advice was provided by [Angjoo Kanazawa's](https://people.eecs.berkeley.edu/~kanazawa/) !
