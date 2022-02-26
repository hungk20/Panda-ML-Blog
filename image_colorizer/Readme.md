Project structure
```
.
├── Readme.md
├── images
│   ├── butterfly.jpeg
│   ├── landscape.jpeg
│   └── portrait.jpeg
├── models
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
├── colorize_image.py
└── utils.py
```

Files in `models` folder are too big for github so please create one & download files from here [download files from here](https://drive.google.com/drive/folders/1FaDajjtAsntF_Sw5gqF0WyakviA5l8-a).

Once it's done, you can use the following command to apply run colorization for a new image:
```
python colorize_image.py --image [path to your image]
```
