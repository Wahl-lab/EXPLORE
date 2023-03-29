# EXPLORE: A novel deep learning-based analysis method for exploration behaviour in object recognition tests

![](EXPLORE.gif)

## :mortar_board: _About:_
Object recognition tests are widely used in neuroscience to assess memory function in rodents. Despite the experimental simplicity of the task, the interpretation of behavioural features that are counted as object exploration can be complicated. Traditionally, analysis of object exploration thus is often based on manually scoring, which is  time-consuming, limited to few behaviours, and variable across researchers. To overcome these limitations We developed "EXLORE", a simple, ready-to use and open source pipeline. Compared to costly commercial software, EXPLORE performs the different analysis steps for object recognition tests with  higher precision, higher versatility and lower time investment. EXPLORE consists of a convolutional neural network trained in a supervised manner, that extracts features from images and classifies behavior of rodents near a presented object as “exploration” or “no exploration". EXPLORE achieves human-level accuracy in identifying and scoring exploration behaviour and outperforms commercial software, in particular under complex conditions, e.g., when multiple objects or larger objects to climb on are present. By labeling the respective training data set, users decide by themselves, which types of interactions are in- or excluded for scoring exploration behaviour. A GUI provides a beginning-to-end analysis with an automatic stop-watch function to calculate the duration of specific exploration behaviour, accelerating a fast and reproducible data analysis for neuroscientists with no expertise in programming or deep learning.

## :hammer: _Install EXPLORE:_

- First install Anaconda (if not installed already): [Install now](https://docs.anaconda.com/anaconda/install/index.html)
- Clone this repository and store the folder *EXPLORE-main* at a preferred directory (first, you find it in your *download* folder)
- Open a shell- or a terminal window and change the directory (the easiest way is to drag & drop your folder into the shell- or terminal window after typing *cd* and a *space*):
```sh
cd <your directory>/EXPLORE-main
```
- create and activate your environment:
```sh
conda create -n XPL
conda activate XPL
```
- Run the *requirements.txt* file (this will install all the necessary packages for EXPLORE into your new conda environment (could take a few minutes!)):
```sh
conda install -c conda-forge --file requirements.txt
```
- install OpenCV with the following command on **macOS**:
```sh
pip install opencv-python==4.1.1.26
```
(use **pip3** for macOS earlier than *BigSur*)

- or install OpenCV with the following command on **Windows**:

```sh
conda install -c conda-forge opencv==4.5.0
```

\
&nbsp;

:fire: **Congratulations, you have now successfully installed EXPLORE! Now let's use it...** :fire:
  
\
&nbsp;
  
## :bulb: _How to use EXPLOREs deep learning-based exploration analysis:_

EXPLOREs deep learning-based exploration analysis is the major part to investigate object recognition tests. There are three parts: 1. Training a network on a few manually scored samples. 2. Predict on all of your experiment videos. 3. Correct your prediction if necessary. The main measures taken are *exploration time* and *exploration frequency* on each defined object. 
:exclamation:Note: For acquisition session and testing session two distinct networks have to be trained.

### Overview on method:
![](https://github.com/victorjonathanibanez/EXPLORE/blob/main/overview_dl.jpg)

\
&nbsp;

Open a shell- or a terminal window and change to your directory:
```sh
cd <your directory>/EXPLORE-main/scripts
```

Activate your virtual environment:
```sh
conda activate XPL
```

### Training:
  
To train a network enter the following command:
```sh
python main_training.py
```
(**python3** for macOS)

\
&nbsp;

**:arrow_right: This will now open a GUI (see [manual training1](https://github.com/victorjonathanibanez/EXPLORE/blob/main/manuals/Manual_training1.jpg) and [manual training2](https://github.com/victorjonathanibanez/EXPLORE/blob/main/manuals/Manual_training2.jpg) and [manual scoring](https://github.com/victorjonathanibanez/EXPLORE/blob/main/manuals/Manual_scoring.jpg) for further instructions!)** 
  
\
&nbsp;

### Prediction:
  
To predict on your experiment videos enter the following command:
```sh
python main_prediction.py
```
(**python3** for macOS)

\
&nbsp;

**:arrow_right: This will now open a GUI (see [manual prediction](https://github.com/victorjonathanibanez/EXPLORE/blob/main/manuals/Manual_prediction.jpg) for further instructions!)** 

\
&nbsp;

### Correction:
  
To correct your prediction enter the following command:
```sh
python main_correct.py
```
(**python3** for macOS)

\
&nbsp;

**:arrow_right: This will now open a GUI (see [manual correction](https://github.com/victorjonathanibanez/EXPLORE/blob/main/manuals/Manual_correction.jpg) for further instructions!)** 

\
&nbsp;

| Output files | Type | Description | 
| ------ | ------ | ------ |
| Prediction videos | folder | For all of the selected experiment videos EXPLORE will generate colored squares around the objects whenever exploration behaviour was predicted and stores the newly created videos in a folder *prediction videos*|
| Dataframe | .csv | The predicted exploration times and frequencies at each object will be stored in a dataframe |
| Plots | .png | Training- and validation accuracy- and loss will be plotted and saved |
  
\
&nbsp;

## :bulb: _How to use EXPLOREs manual labeling tool:_
Besides the automated analysis, EXPLORE provides a tool for manual scoring. The scoring will be saved as .csv file.

Open a shell- or a terminal window and change to your directory:
```sh
cd <your directory>/EXPLORE-main/scripts
```

Activate your virtual environment:
```sh
conda activate XPL
```

To start manual scoring type the following command:
```sh
python main_manual_scoring.py
```
(**python3** for macOS)

\
&nbsp;

**:arrow_right: This will now open a GUI (refer to the training manual for further instructions!)** 

\
&nbsp;

## :bulb: _How to use EXPLOREs quadrant analysis:_

With the quadrant analysis you can investigate and quantify movement throughout the experiment arena. Two measures are taken: the time animals spent in each quadrant over a given period (*exploration time*) and the frequency of transistions from one quadrant to another (*exploration frequency*).

### Overview on method:
![](https://github.com/victorjonathanibanez/EXPLORE/blob/main/overview_quadrant.jpg)

\
&nbsp;

Open a shell- or a terminal window and change to your directory:
```sh
cd <your directory>/EXPLORE-main/scripts
```

Activate your virtual environment:
```sh
conda activate XPL
```

Then enter the following command:
```sh
python main_quadrant.py
```
(**python3** for macOS)

\
&nbsp;

**:arrow_right: This will now open a GUI (see [manual quadrant](https://github.com/victorjonathanibanez/EXPLORE/blob/main/manuals/Manual_quadrant.jpg) for further instructions!)** 

\
&nbsp;

| Output files | Type | Description | 
| ------ | ------ | ------ |
| Dataframe | .csv | The predicted exploration times and frequencies for each quadrant will be stored in a dataframe |
| Plots | .png | For each animal (video) the frequency will be plotted and stored |
| Heatmap | .png | An overview on the quadrants exploration- frequency and time will be plotted as heatmaps |

\
&nbsp;

\
&nbsp;

:exclamation: **Please refer to our publication for further information about more technical details: https://www.nature.com/articles/s41598-023-31094-w**

\
&nbsp;

## :mailbox: _Contact:_
victor.ibanez@uzh.ch\
wahl@hifo.uzh.ch
