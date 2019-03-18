
import rdkit
import pandas as pd
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

#%%
dir_lincslib='/Users/nienke/Dropbox/Harvard/CBDM-SORGER/Collaborations/Patrick_Nienke/screening_results/input'
dir_broad_ids='/Users/nienke/Dropbox/Harvard/CBDM-SORGER/Collaborations/Patrick_Nienke/screening_results/output'

#os.chdir(dir_lincslib)
os.chdir(dir_broad_ids)
print(os.listdir())
#%%
os.chdir(dir_broad_ids)
list_broad=pd.read_csv('cmpd_info_COCA_PDX_20170629_compact.csv')

#%%
os.chdir(dir_lincslib)
list_lincs=pd.read_csv('reagenttracker_small_molecules_20170629.csv')

#%%

def make_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
              
#%% get fingerprint chembl database

fps_broad=[]
i=-1

for rec in list_broad.iterrows():
    try:
        i=i+1
        a=[rec[1][0],make_fingerprint(
                Chem.MolFromInchi(
                Chem.MolToInchi(
                Chem.MolFromSmiles(rec[1][1])
                ))
                )]
        fps_broad.append(a)
        print(i)
    except Exception:
        continue 

#%%get fingerprint lincs molecules     

fps_lincs=[]
j=-1

for rec in list_lincs.iterrows():
    try:
        j=j+1
        a=[rec[1][0],make_fingerprint(Chem.MolFromInchi(rec[1][3]))]
        fps_lincs.append(a)
        print(j)
    except Exception:
        continue 
    
#%% calculate similarity

matches=[]

for rec_1 in enumerate(fps_broad):
    best_similarity=0
    best_names=[]
    for rec_2 in enumerate(fps_lincs):
        similarity=DataStructs.FingerprintSimilarity(
                                                     rec_1[1][1], rec_2[1][1], DataStructs.TanimotoSimilarity
                                                     )
        #print(qi,di)
        if similarity > best_similarity:
            best_similarity = similarity
            best_names =[rec_2[1][0]]
        elif similarity==best_similarity:
            best_names.append([rec_2[1][0]])
    print(rec_1[0])
    for m in best_names:
        matches.append([rec_1[1][0],m,best_similarity])
#%%
matches_df=pd.DataFrame(matches)
matches_df.columns=['broad_id','hms_id','similarity']

matches_df.loc[matches_df.similarity<1]

#data[data[:,0] == 100002]
        
#%%

os.chdir(dir_broad_ids)
matches_df.to_csv('broadID_2_hmsID_similarities.csv', index=False)

#%%





















    
    
    
    