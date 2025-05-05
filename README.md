# House Sale Price Prediction Project

This project involves creating a regression model to predict house sale prices based on a variety of features. Below is a detailed description of the dataset features used in the analysis.

---

## Dataset Features

### General Information
- **MSSubClass**: Type of dwelling involved in the sale.
    - `20`: 1-Story 1946 & Newer All Styles
    - `30`: 1-Story 1945 & Older
    - `40`: 1-Story w/Finished Attic All Ages
    - `45`: 1-1/2 Story - Unfinished All Ages
    - `50`: 1-1/2 Story Finished All Ages
    - `60`: 2-Story 1946 & Newer
    - `70`: 2-Story 1945 & Older
    - `75`: 2-1/2 Story All Ages
    - `80`: Split or Multi-Level
    - `85`: Split Foyer
    - `90`: Duplex - All Styles and Ages
    - `120`: 1-Story PUD (Planned Unit Development) - 1946 & Newer
    - `150`: 1-1/2 Story PUD - All Ages
    - `160`: 2-Story PUD - 1946 & Newer
    - `180`: PUD - Multilevel - Incl Split Lev/Foyer
    - `190`: 2 Family Conversion - All Styles and Ages

- **MSZoning**: General zoning classification of the sale.
    - `A`: Agriculture
    - `C`: Commercial
    - `FV`: Floating Village Residential
    - `I`: Industrial
    - `RH`: Residential High Density
    - `RL`: Residential Low Density
    - `RP`: Residential Low Density Park
    - `RM`: Residential Medium Density

- **LotFrontage**: Linear feet of street connected to property.
- **LotArea**: Lot size in square feet.

---

### Property Access and Shape
- **Street**: Type of road access to property.
    - `Grvl`: Gravel
    - `Pave`: Paved

- **Alley**: Type of alley access to property.
    - `Grvl`: Gravel
    - `Pave`: Paved
    - `NA`: No alley access

- **LotShape**: General shape of property.
    - `Reg`: Regular
    - `IR1`: Slightly irregular
    - `IR2`: Moderately irregular
    - `IR3`: Irregular

- **LandContour**: Flatness of the property.
    - `Lvl`: Near Flat/Level
    - `Bnk`: Banked - Quick and significant rise from street grade to building
    - `HLS`: Hillside - Significant slope from side to side
    - `Low`: Depression

- **Utilities**: Type of utilities available.
    - `AllPub`: All public utilities (Electricity, Gas, Water, & Sewer)
    - `NoSewr`: Electricity, Gas, and Water (Septic Tank)
    - `NoSeWa`: Electricity and Gas Only
    - `ELO`: Electricity only

- **LotConfig**: Lot configuration.
    - `Inside`: Inside lot
    - `Corner`: Corner lot
    - `CulDSac`: Cul-de-sac
    - `FR2`: Frontage on 2 sides of property
    - `FR3`: Frontage on 3 sides of property

- **LandSlope**: Slope of property.
    - `Gtl`: Gentle slope
    - `Mod`: Moderate slope
    - `Sev`: Severe slope

---

### Neighborhood and Proximity
- **Neighborhood**: Physical locations within Ames city limits.
    - Examples: `Blmngtn`, `Blueste`, `BrDale`, `BrkSide`, `ClearCr`, etc.

- **Condition1**: Proximity to various conditions.
    - Examples: `Artery`, `Feedr`, `Norm`, `RRNn`, `RRAn`, etc.

- **Condition2**: Proximity to various conditions (if more than one is present).
    - Examples: `Artery`, `Feedr`, `Norm`, `RRNn`, `RRAn`, etc.

---

### Building Characteristics
- **BldgType**: Type of dwelling.
    - `1Fam`: Single-family Detached
    - `2FmCon`: Two-family Conversion
    - `Duplx`: Duplex
    - `TwnhsE`: Townhouse End Unit
    - `TwnhsI`: Townhouse Inside Unit

- **HouseStyle**: Style of dwelling.
    - Examples: `1Story`, `1.5Fin`, `2Story`, `SFoyer`, `SLvl`, etc.

- **OverallQual**: Rates the overall material and finish of the house (1-10 scale).
- **OverallCond**: Rates the overall condition of the house (1-10 scale).

- **YearBuilt**: Original construction date.
- **YearRemodAdd**: Remodel date (same as construction date if no remodeling or additions).

---

### Additional Features
- **RoofStyle**: Type of roof.
    - Examples: `Flat`, `Gable`, `Hip`, etc.

- **Exterior1st** and **Exterior2nd**: Exterior covering on house.
    - Examples: `VinylSd`, `HdBoard`, `Stucco`, etc.

- **Foundation**: Type of foundation.
    - Examples: `BrkTil`, `CBlock`, `PConc`, etc.

- **Heating**: Type of heating.
    - Examples: `GasA`, `GasW`, `Wall`, etc.

- **CentralAir**: Central air conditioning (`Y` for Yes, `N` for No).

- **GarageType**: Garage location.
    - Examples: `Attchd`, `Detchd`, `BuiltIn`, etc.

- **PoolQC**: Pool quality.
    - Examples: `Ex`, `Gd`, `TA`, `Fa`, `NA`.

- **Fence**: Fence quality.
    - Examples: `GdPrv`, `MnPrv`, `GdWo`, `MnWw`, `NA`.

---

This README provides a comprehensive overview of the dataset features used in the Kaggle competition project for predicting house sale prices. For further details, refer to the dataset documentation.


# Installing libraries
install.packages("systemfonts")

# Install basic system dependencies first
install.packages(c("httr", "xml2", "curl", "gargle"), dependencies = TRUE)

install.packages(c("tidyverse", "caret", "readr", "Metrics", "glmnet", "randomForest", "xgboost"))
install.packages("randomForest") 
install.packages("xgboost")
install.packages("data.table") 
install.packages("glmnet")
install.packages("Metrics")
install.packages("tidyverse")
install.packages("textshaping")
install.packages("ragg")

install.packages("corrplot")
#install.packages("nloptr")
install.packages("nloptr", configure.args = "--with-nlopt=/usr")
install.packages("mice")

# to save plots in pdf or PNG
install.packages("patchwork")
install.packages("gridExtra")
#install.packages("reshape2")
library(reshape2)

library(data.table)
library(ggplot2)
library(systemfonts)
library(textshaping)
library(mice)
library(ragg)
library(tidyverse)
library(caret)
library(readr)
library(Metrics)
library(glmnet)
library(randomForest)
library(data.table)

library(patchwork)
library(gridExtra)
library(corrplot)

# Top Features Correlated with SalePrice
chart link: https://rpubs.com/laurindocbenjam/predict-house-sale-price-in-R
