# Impact of Design Decisions in Scanpath Modeling
1. PARVIN EMAMI
2. YUE JIANG and ZIXIN GUO
3. LUIS A. LEIVA

Modeling visual saliency in graphical user interfaces (GUIs) is important to understand how people perceive GUI designs
and what elements attract their attention. One aspect that is often overlooked is the fact that computational models 
depend on a series of design parameters that are not straightforward to decide. We systematically analyze how different 
design parameters affect scanpath evaluation metrics using a state-ofthe-art computational model (DeepGaze++). 
We particularly focus on three design parameters: input image size, inhibition-of-return decay, and masking radius. 
We show that even small variations of these design parameters have a noticeable impact on standard evaluation metrics 
such as DTW or Eyenalysis. These effects also occur in other scanpath models, such as UMSS and ScanGAN, and in other 
datasets such as MASSVIS. Taken together, our results put forward the impact of design decisions for predicting 
users’ viewing behavior on GUIs.

Link to the paper: [Impact of Design Decisions in Scanpath Modeling](https://dl.acm.org/doi/10.1145/3655602)

# install libraries

```
pip install -r requirements.txt
```

you may need to create a new venv.

# Data
The UEyes dataset is available [here](https://userinterfaces.aalto.fi/ueyeschi23/).
\
The MASSVIS dataset is available [here](http://massvis.mit.edu).

# run

you can run the grid search experiments using the different parameters. 
`` 

**NOTE**: main script generates fixations for the given images and saves the results in a csv file.


main function has the following parameters:
```
  -i, --image_folders PATH        Containing images
  -c, --csv_folders PATH          Containing one csv per user with fixations
  -d, --device TEXT               Device to use for training
  -n, --number_of_fixations INT   Number of fixations to generate, if zero it
                                  will be same as the number of fixations in GT
  -w, --width INT                 Width of the image
  -h, --height INT                Height of the image
  -r, --radius FLOAT              Radius of the masking it should be a float
                                  between 0 and 1
  --gamma FLOAT                   masking parameter or IOR decay
  -o, --output PATH               Output file to save the results, it can
                                  be json or csv (if not provided, it will not
                                  be saved)
  -m, --max-number-of-images INT
                                  Maximum number of images to process (for
                                  debugging purposes) (default: -1, all
                                  images)
  --noise TEXT                    Noise to add to the fixation
  
  -t, --images-category PATH      it should be a csv file with two columns:
                                  `Image Name` and `Category`
  -k, --mask-type TEXT            can be one of these: "new, "old_circle"

```
for instance for UEyes data run with the best parameters:
```bash
python main.py main    -i ../UEyes_dataset/test/images \
                       -c ../UEyes_dataset/test/eyetracker_logs \
                       -n 10 \
                       -w 128 \
                       -h 128 \
                       -r 0.05 \
                       --gamma 0.1 \
                       --noise ZERO \
                       --device cuda:0 \
                       --mask-type new \
                       --output test.csv \
                       --images-category ../UEyes_dataset/image_types.csv
```
