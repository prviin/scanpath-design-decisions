# install libraries

```
pip install -r requirements.txt
```

you may need to create a new venv.

# Data
the dataset is available [here](https://zenodo.org/record/8010312).
the final folder structure is like:
```
saliency_modeling_issues
├── Deepgaze++
│   ├── main.py
│   │   
│   │   
│   │   
│   ├── data
│   └── ...
└── UEyes_dataset
    ├── eyetracker_logs
    ├── images
    ├── image_types.csv
    └── ...
```
# run

## run grid search experiments
you can run the grid search experiments using the `run_statistical.sh`. 

**NOTE**: This script generates fixations for the given images and

Options:
  -i, --image_folders PATH
  -c, --csv_folders PATH
  -d, --device TEXT               Device to use for training
  -n, --number_of_fixations INTEGER
                                  Number of fixations to generate, if zero it
                                  will be same as the number of fixations in
                                  GT
  -w, --width INTEGER             Width of the image
  -h, --height INTEGER            Height of the image
  -r, --radius FLOAT              Radius of the fixation it should be a float
                                  between 0 and 1
  --gamma FLOAT                   masking parameter
  -o, --output PATH               Output file to save the results64x64, it can
                                  be json or csv (if not provided, it will not
                                  be saved)
  -m, --max-number-of-images INTEGER
                                  Maximum number of images to process (for
                                  debugging purposes) (default: -1, all
                                  images)
  --noise                         ZERO|GAUSSIAN|SALT_AND_PEPPER|POISSON|NOISE
  -t, --images-category PATH      it should be a csv file with two columns:
                                  `Image Name` and `Category`
  -k, --mask-type TEXT            can be one of these: "new, "old_circle"


for instance for new data run:

```
python main.py -i ../UEyes_dataset/images \
               -c ../UEyes_dataset/eyetracker_logs \
               -n 10 \
               -w 225 \
               -h 225 \
               -r 0.2 \
               --gamma 0.1 \
               --noise ZERO \
               --device cuda:0 \
               --mask-type old_circle \
               --output test.csv \
               --images-category ../UEyes_dataset/image_types.csv
```
