##  Identification of compounding drivers of river floods
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.7765151-blue.svg)](https://doi.org/10.5281/zenodo.7765151)

### Overview
The repository contains the codes to reproduce the main procedure for identifying the compounding drivers of river floods and the associated flood complexity, and the data used for producing the main figures in the paper:

> Jiang *et al.*, **The importance of compounding drivers for large river floods** (*Submitted to Journal*)

Description of columns in `results.csv`: 
> - lat: Latitude coordinates of station (*released after acceptance*)
> - long: Longitude coordinates of station (*released after acceptance*)
> - prop_rr: Proportion of recent rainfall as a main driver of AM floods
> - prop_tg: Proportion of recent temperature as a main driver of AM floods
> - prop_sm: Proportion of soil moisture as a main driver of AM floods
> - prop_sp: Proportion of snowpack as a main driver of AM floods
> - prop_mu: Proportion of multi-driver floods
> - mag_ratio: Magnitude ratio of multi-driver floods to single-driver floods
> - mag_ttest_p: T-test p-value for the mean magnitude of multi-driver floods vs. single-driver floods
> - flood_com: Flood complexity
> - flood_com_p: Combined p-value for the flood complexity slope
> - est_err: Estimation error in the magnitude of the largest observed floods

Please refer to the file [LICENSE](https://github.com/oreopie/hydro-interpretive-dl/blob/main/LICENSE) for the license governing this code.

Kindly contact us with any questions or ideas you have concerning the code, or if you discover a bug. You may raise an issue [here](https://github.com/oreopie/flood-compounding-drivers/issues) or contact Shijie Jiang through email at shijie.jiang(at)hotmail.com
