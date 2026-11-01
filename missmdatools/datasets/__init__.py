# -*- coding: utf-8 -*-
from __future__ import annotations
import pathlib
from pandas import read_excel
from pyreadr import read_r

DATASETS_DIR = pathlib.Path(__file__).parent / "data"

#------------------------------------------ adult census income dataset ----------------------------------------------------- 
adult = read_excel(DATASETS_DIR/"adult-census-income.xlsx")
adult.__doc__ = """
Adult Census Income

Predict whether annual income of an individual exceeds $50K/yr based on census data. Also known as "Census Income" dataset.
Prediction task is to determine whether a person's income is over $50,000 a year.

Data with 32561 rows and 15 columns.

Columns:
    1. age: continuous.
    2. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    3. fnlwgt: continuous.
    4. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 
        1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    5. education-num: continuous.
    6. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    7. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, 
        Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    8. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    9. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    10. sex: Female, Male.
    11. capital-gain: continuous.
    12. capital-loss: continuous.
    13. hours-per-week: continuous.
    14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, 
        Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, 
        Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, 
        Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    15. income (target) : >50K, <=50K.

References
----------
UCI dataset https://archive.ics.uci.edu/dataset/2/adult
    
Examples
--------
>>> from missmdatools.datasets import adult
>>> from missmdatools import FAMDimputer
>>> clf = FAMDimputer()
>>> clf.fit(adult)
FAMDimputer()
"""

#------------------------------------------ children dataset ----------------------------------------------------- 
children = read_r(DATASETS_DIR/'children.rda')["children"]
children = children.astype("float")
children.__doc__ = """
Children Dataset

The data used here is a contingency table that summarizes the answers given by different categories of people to the following question : 
according to you, what are the reasons that can make hesitate a woman or a couple to have children?
A data frame with 18 rows and 9 columns:

    * Rows represent the different reasons mentioned, 
    * Columns represent the different categories (education, age) people belong to.

Examples
--------
>>> from missmdatools.datasets import children
>>> from missmdatools import CAimputer
>>> clf = CAimputer()
>>> clf.fit(children)
CAimputer()
"""

#------------------------------------------ credit dataset ----------------------------------------------------- 
credit = read_excel(DATASETS_DIR/"credit-approval.xlsx")
credit.__doc__ =  """
Credit Approval

This data concerns credit card applications; good mix of attributes.
Data with 690 rows and 16 columns.

Columns:
    1. A1:	b, a.
    2. A2:	continuous.
    3. A3:	continuous.
    4. A4:	u, y, l, t.
    5. A5:	g, p, gg.
    6. A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    7. A7:	v, h, bb, j, n, z, dd, ff, o.
    8. A8:	continuous.
    9. A9:	t, f.
    10. A10: t, f.
    11. A11: continuous.
    12. A12: t, f.
    13. A13: g, p, s.
    14. A14: continuous.
    15. A15: continuous.
    16. A16 (target): positive,negative (class attribute)

References
----------
UCI dataset https://archive.ics.uci.edu/dataset/27/credit+approval

Examples
--------
>>> from missmdatools.datasets import credit
>>> from missmdatools import FAMDimputer
>>> clf = FAMDimputer()
>>> clf.fit(credit)
FAMDimputer()
"""

#------------------------------------------ disease dataset ----------------------------------------------------- 
disease = read_excel(DATASETS_DIR/"heart-disease.xlsx")
disease.__doc__ = """
Cleveland Heart Disease

Data with 303 rows and 14 columns.

Columns:
    1. age (in years) : integer
    2. sex: male (1), female (0)    
    3. cp (chest pain type): typical angina (1), atypical angina (2), non-anginal pain (3), asymptomatic (4) 
    4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
    5. chol (serum cholestoral in mg/dl): integer   
    6. fbs (fasting blood sugar > 120 mg/dl) : true (1), false (0)  
    7. restecg (resting electrocardiographic results) : 
        * 0: normal
        * 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        * 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
    8. thalach (maximum heart rate achieved)
    9. exang (exercise induced angina) : yes (1), no (0)
    10. oldpeak (ST depression induced by exercise relative to rest)
    11. slope (the slope of the peak exercise ST segment):
        * 1: upsloping
        * 2: flat
        * 3: downsloping
    12. ca (number of major vessels (0-3) colored by flourosopy)    
    13. thal : 3 = normal; 6 = fixed defect; 7 = reversable defect 
    14. heart-disease (target) : absence, presence

References
----------
UCI dataset https://archive.ics.uci.edu/dataset/45/heart+disease

Examples
--------
>>> from missmdatools.datasets import disease
>>> from missmdatools import FAMDimputer
>>> clf = FAMDimputer()
>>> clf.fit(disease)
FAMDimputer()
"""

#------------------------------------------ orange dataset ----------------------------------------------------- 
orange = read_r(DATASETS_DIR/'orange.rda')["orange"]
orange.__doc__ = """
Sensory description of 12 orange juices by 8 attributes

A data frame with 12 rows and 8 columns. Rows represent the different orange juices, columns represent the attributes.
Some values are missing.

Examples
--------
>>> from missmdatools.datasets import orange
>>> from missmdatools import PCAimputer
>>> clf = PCAimputer()
>>> clf.fit(orange)
PCAimputer()
"""

#------------------------------------------ ozone dataset ----------------------------------------------------- 
ozone = read_r(DATASETS_DIR/'ozone.rda')["ozone"]
columns = [x for x in ozone.columns if x not in ("vent","pluie")]
for x in columns:
    ozone[x] = ozone[x].astype("float")
ozone.__doc__ = """
Daily measurements of meteorological variables and ozone concentration

This dataset contains 112 daily measurements of meteorological variables (wind speed, temperature, rainfall, etc.) 
and ozone concentration recorded in Rennes (France) during the summer 2001. 
There are 11 continuous variables and 2 categorical variables with 2 or 4 levels. Some values are missing.

Examples
--------
>>> from missmdatools.datasets import ozone
>>> from missmdatools import FAMDimputer
>>> clf = FAMDimputer()
>>> clf.fit(ozone)
FAMDimputer()
"""

#------------------------------------------ subvoting dataset ----------------------------------------------------- 
subvoting = read_excel(DATASETS_DIR/"subset_voting_records.xlsx").astype("category")
subvoting.__doc__ = """
Subset of Congressional Voting Records

1984 United Stated Congressional Voting Records; Classify as Republican or Democrat

This data set includes votes for each of the U.S. House of Representatives Congressmen on 
the 16 key votes identified by the CQA.  The CQA lists nine different types of votes: voted for, 
paired for, and announced for (these three simplified to yea), voted against, paired against, 
and announced against (these three simplified to nay), voted present, voted present to avoid 
conflict of interest, and did not vote or otherwise make a position 
known (these three simplified to an unknown disposition).

Columns:
    1. Class Name: 2 (democrat, republican)
    2. handicapped-infants: 2 (y,n)
    3. water-project-cost-sharing: 2 (y,n)
    4. adoption-of-the-budget-resolution: 2 (y,n)
    5. physician-fee-freeze: 2 (y,n)
    6. el-salvador-aid: 2 (y,n)
    7. religious-groups-in-schools: 2 (y,n)
    8. anti-satellite-test-ban: 2 (y,n)
    9. aid-to-nicaraguan-contras: 2 (y,n)
    10. mx-missile: 2 (y,n)
    11. immigration: 2 (y,n)
    12. synfuels-corporation-cutback: 2 (y,n)
    13. education-spending: 2 (y,n)
    14. superfund-right-to-sue: 2 (y,n)
    15. crime: 2 (y,n)
    16. duty-free-exports: 2 (y,n)
    17. export-administration-act-south-africa: 2 (y,n)
    
Examples
--------
>>> from missmdatools.datasets import subvoting
>>> from missmdatools import MCAestimncp
>>> clf = MCAestimncp()
>>> clf.fit(subvoting)
MCAestimncp()
"""

#------------------------------------------ vnf dataset ----------------------------------------------------- 
vnf = read_r(DATASETS_DIR/'vnf.rda')["vnf"]
vnf.__doc__ = """
Questionnaire done by 1232 individuals who answered 14 questions

A user satisfaction survey of pleasure craft operators on the “Canal des Deux Mers”, located in South of France, 
was carried out by the public corporation “Voies Navigables de France” responsible for managing and developing 
the largest network of navigable waterways in Europe. Some values are missing.

Examples
--------
>>> from missmdatools.datasets import vnf
>>> from missmdatools import MCAimputer
>>> clf = MCAimputer()
>>> clf.fit(vnf)
MCAimputer()
"""

#------------------------------------------ shuttle dataset ----------------------------------------------------- 
voting = read_excel(DATASETS_DIR/"voting_records.xlsx").astype("category")
voting.__doc__ = """
Congressional Voting Records

1984 United Stated Congressional Voting Records; Classify as Republican or Democrat

This data set includes votes for each of the U.S. House of Representatives Congressmen on 
the 16 key votes identified by the CQA.  The CQA lists nine different types of votes: voted for, 
paired for, and announced for (these three simplified to yea), voted against, paired against, 
and announced against (these three simplified to nay), voted present, voted present to avoid 
conflict of interest, and did not vote or otherwise make a position 
known (these three simplified to an unknown disposition).

Columns:
    1. Class Name: 2 (democrat, republican)
    2. handicapped-infants: 2 (y,n)
    3. water-project-cost-sharing: 2 (y,n)
    4. adoption-of-the-budget-resolution: 2 (y,n)
    5. physician-fee-freeze: 2 (y,n)
    6. el-salvador-aid: 2 (y,n)
    7. religious-groups-in-schools: 2 (y,n)
    8. anti-satellite-test-ban: 2 (y,n)
    9. aid-to-nicaraguan-contras: 2 (y,n)
    10. mx-missile: 2 (y,n)
    11. immigration: 2 (y,n)
    12. synfuels-corporation-cutback: 2 (y,n)
    13. education-spending: 2 (y,n)
    14. superfund-right-to-sue: 2 (y,n)
    15. crime: 2 (y,n)
    16. duty-free-exports: 2 (y,n)
    17. export-administration-act-south-africa: 2 (y,n)
    
Examples
--------
>>> from missmdatools.datasets import voting
>>> from missmdatools import MCAestimncp
>>> clf = MCAestimncp()
>>> clf.fit(voting)
MCAestimncp()
"""

__all__ = [
    "adult",
    "children",
    "credit",
    "disease",
    "orange",
    "ozone",
    "subvoting",
    "vnf",
    "voting"
]