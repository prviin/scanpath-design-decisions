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

**NOTE**: the command line address should be in the `Deepgaze++` folder
you can run the `python main.py --help` to see the arguments' description.

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
