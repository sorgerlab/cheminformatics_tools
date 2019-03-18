#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 20:04:25 2016

@author: nienke
"""
#%%

import rdkit
import pandas as pd
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

dir_input='/Users/nienke/Dropbox/Harvard/CBDM-SORGER/Collaborations/LINCS_Compound_Database_NM/Compare_Libraries/2017_combine_libs'
dir_output='/Users/nienke/Dropbox/Harvard/CBDM-SORGER/Collaborations/LINCS_Compound_Database_NM/Compare_Libraries/2017_combine_libs/chemical_similarity'

os.chdir(dir_input)

#%%

def make_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)



#%%
lib_combined_csv=pd.read_csv('libraries_combined_20170301_tidy.csv')
lib_combined_csv_t=lib_combined_csv.iloc[0:10]

#%%

fps_db=[]

for rec in lib_combined_csv_t.iterrows():
    try:
        a=[rec[1][0],
                 make_fingerprint(Chem.MolFromInchi(rec[1][2]))]
        fps_db.append(a)
    except Exception:
        continue
    
#%%
matches_trial=[]
for rec_1 in enumerate(fps_db):
    for rec_2 in enumerate(fps_db):
        similarity=DataStructs.FingerprintSimilarity(
                                                     rec_1[1][1], rec_2[1][1], DataStructs.TanimotoSimilarity
                                                     )
        matches_trial.append([rec_1[1][0],rec_2[1][0],similarity])
    print(rec_1[1][0])


        #%%
matches_df=pd.DataFrame(matches_trial)
matches_df.columns=['query','match','similarity']

# matches_df.to_csv('csv_export_similarities_LINCStoCHEMBL_complete.csv', index=False)
