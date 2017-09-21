# README

## Identifying Green Rooftops in Rotterdam to Improve Urban Planning

### Overview

DSSG has partnered with the Municipality of Rotterdam to identify green rooftops in Rotterdam for a given Area of Interest (AoI). In our case, our AoI was the city center.AoI). In our case, our AoI was the city center.AoI). In our case, our AoI was the city center.AoI). In our case, our AoI was the city center.

To achieve this goal, we build a multi-class classification model for rooftops in Rotterdam, classifying them in ‘non-vegetation’, ‘vegetation,’ and ‘trees.’ Our model is used to produce three deliverables for the Municipality of Rotterdam:

1. a ‘Raw Table’ with the predicted classification for each rooftop in AoI, including its prediction probability and corresponding area in square meters;corresponding area in square meters;

2. a ‘Summary Table,’ which is a ‘Raw Table’ grouped by type of classification; information is displayed in by area and by rooftop; and,

3. a ‘Visualization’ that identifies the predicted classifications in a map of the AoI with colored shapefiles for each building footprint (i.e., grey, light green, and dark green for non-vegetation, vegetation, and trees, respectively); this file can be opened with QGIS, a free and open source geographic information system software.

### Installation

To set up a Python environment, go to https://conda.io/miniconda.html. Choose the Miniconda installer (Python 3.6) according to your operating system.

Once Miniconda is installed, use the ‘conda’ command to create an environment called ‘green-roofs’ with the file ‘environment.yml’.  When you create an environment or install package, conda will ask you to proceed (i.e., ‘proceed ([y]/n)?‘). Type ‘y’.

To create the environment and install the necessary packages, open the terminal inside folder '0_install' and type:
- For Windows: ‘conda env create -f win_environment.yml -n greenroofs’;
- For Linux/macOS: ‘conda env create -f unix_environment.yml -n greenroofs’.

To activate the environment, type ‘activate greenroofs’ (Windows) or ‘source activate greenroofs’ (Linux/macOS). To deactivate the environment, type ‘deactivate’ (Windows) or ‘source deactivate’ (Linux/macOS).

You now have a working Python environment.

### Instructions

1. Open the terminal and activate the environment;

2. Open ‘create_features.py’:
    
    1. Set ‘BAG_Panden’ to the file with the building footprint polygons;
    2. Set ‘CIR’ to the file with the color-infrared imagery;
    3. Save ‘create_features.py’;

3. Run ‘create_features.py’ (i.e., type ‘python create_features.py’ in the command line);

4. Open ‘classify.py’ and:

    1. Save the IDs within the selected area of interest as ‘aoi_ids.csv’ and set it as the variable ‘AoI’;
    2. Save ‘classify.py’; and,

5. Run ‘classify.py’ (i.e., type ‘python classify.py in the command line’).
