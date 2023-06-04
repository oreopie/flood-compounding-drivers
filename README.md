##  Identification of compounding drivers of river floods
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.7765151-blue.svg)](https://doi.org/10.5281/zenodo.7765151)

This repository contains the code to reproduce the main procedure for identifying the compounding drivers of river floods and the associated flood complexity, as presented in the paper:

> Jiang *et al.*, **Compounding effects in flood drivers challenge estimates of extreme river floods** (*Submitted to Journal*)

## Overview

The aim of this research is to investigate the compounding effects of various drivers on river floods and to understand their influence on flood severity and estimates of extreme floods. The code and data provided in this repository allow for the reproduction of the main analyses and figures presented in the paper.

The repository is structured as follows:
                                          
```
|-- data/
|   |-- sample.csv                          # Demo data for a river basin
|-- libs/                                   # Custom functions (released after acceptance)
|   |-- plots.py
|   |-- utils.py
|-- outputs/                                # Folder to save the output to
|-- analyze_individual_catchment.ipynb      # Jupyter Notebook for the demo to obtain Figs. S4 and 4A
|-- requirements.txt                        # PyPI dependencies
|-- results.csv                             # Main results for all catchments
|-- run.py                                  # Standalone script to obtain results (released after acceptance)
```

##  Quick Start

The code was tested with Python 3.8 (on Windows 10/11 and macOS Ventura). 
To use this code please do the following in command line:

a) Change into this directory

`cd /path/to/flood-compounding-drivers`

b) Install dependencies (`conda`/`virtualenv` is recommended to manage packages):

`pip install -r ./requirements.txt`

*It may take ~30 mins to run the script (100 replicates of 5-fold cross-validation).*

c) Start Jupyter Notebook and run `analyze_individual_catchment.ipynb` in the browser:

`jupyter notebook`

d) Alternatively, run the standalone script to get the results:

`python run.py --input_path=./data/sample.csv --basin_size=827.00 --output_dir=./outputs/`

##  Description of `results.csv` (used to generate Figs. 2-5 in the text)

Description of columns in `results.csv`:

```
- lat:         Latitude coordinates of station (*released after acceptance*)
- long:        Longitude coordinates of station (*released after acceptance*)
- prop_rr:     Proportion of recent rainfall as a main driver of AM floods
- prop_tg:     Proportion of recent temperature as a main driver of AM floods
- prop_sm:     Proportion of soil moisture as a main driver of AM floods
- prop_sp:     Proportion of snowpack as a main driver of AM floods
- prop_mu:     Proportion of multi-driver floods
- mag_ratio:   Magnitude ratio of multi-driver floods to single-driver floods
- mag_ttest_p: T-test p-value for the mean magnitude of multi-driver floods vs. single-driver floods
- flood_com:   Flood complexity
- flood_com_p: Combined p-value for the flood complexity slope
- est_err:     Estimation error in the magnitude of the largest observed floods
```

## Prepare the dataset

The dataset in the study is implemented based on the following data and codes:

- River discharge and gauge location: https://www.bafg.de/GRDC
- Global daily precipitation: http://www.gloh2o.org/mswep
- Global daily temperature: http://www.gloh2o.org/mswx
- The HBV model and the associated parameter maps used to generate daily soil moisture and snowpack: http://www.gloh2o.org/hbv/
- Catchment static attributes: https://www.hydrosheds.org/hydroatlas
- Catchment boundary delineation: https://github.com/xiejx5/watershed_delineation

After all grid-based data are prepared, the catchment average data are then calculated by the tool: https://github.com/nzahasan/pyscissor

## Contact Information

For any questions or inquiries about this research or repository, please contact the corresponding author:

- Name: Shijie Jiang
- Email: shijie.jiang(at)hotmail.com

## License

This project is licensed under the [MIT License](LICENSE).

When using the code from this repository, we kindly request that you cite the paper.

Thank you for your interest in our work!
