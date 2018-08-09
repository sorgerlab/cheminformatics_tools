
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from IPython.display import display
from ipywidgets import HBox, VBox, Box, Label, Image, HTML
import re
import copy
import functools
import requests
import pandas as pd
import getpass
import numpy as np
import matplotlib.pyplot as plt
import ipyparallel
from tqdm import tqdm, tqdm_notebook


# In[3]:


from rdkit.Chem import MolFromSmiles, Draw, rdFMCS
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.ChemUtils.AlignDepict import AlignDepict
from rdkit.DataStructs import DiceSimilarity
from rdkit.Chem import AllChem
import molvs


# # Set up parallel execution

# In[4]:


parallel_client = ipyparallel.Client()
pview = parallel_client.load_balanced_view()


# In[5]:


def parallel_progress_map(f, data, chunksize=50):
    result = pview.map(f, data, chunksize=100, ordered=False)
    for _ in tqdm_notebook(result, total=len(data)):
        pass
    return list(result)


# # Existing compounds (ReagentTracker)

# ## Load the current small molecule list from ReagentTracker

# First we prompt the user for their Ecommons login so we can authenticate with RT.

# In[6]:


username = input('Username: ')
password = getpass.getpass('Password: ')


# In[7]:


r = requests.post('https://reagenttracker.hms.harvard.edu/api/v0/login',
                  json={'username': username, 'password': password})
cookies = r.cookies
rt_response = requests.get('https://reagenttracker.hms.harvard.edu/api/v0/search?q=', cookies=cookies).json()


# Build a DataFrame from the small molecule data in the API results.

# In[8]:


sm = [d for d in rt_response['canonicals'] if d['type'] == 'small_molecule' and d['name'] != 'DEPRECATED']
rt_data = pd.DataFrame(sm)[['lincs_id', 'name', 'alternate_names', 'smiles']]
rt_data.set_index('lincs_id', inplace=True, verify_integrity=True)
rt_data.rename(columns={'smiles': 'smiles_original'}, inplace=True)
rt_data['smiles_original'].fillna('', inplace=True)


# Generate canonical tautomers.

# In[9]:


canonicalize_tautomers = molvs.tautomer.TautomerCanonicalizer(max_tautomers=20)


# In[10]:


rt_data['mol'] = parallel_progress_map(canonicalize_tautomers, rt_data['smiles_original'].map(MolFromSmiles))


# Generate isomeric SMILES for later merging with new compound records.

# In[11]:


rt_data['smiles'] = rt_data['mol'].map(lambda x: AllChem.MolToSmiles(x, isomericSmiles=True))


# # New compounds (Bhola screen)

# ## Load the structure file containing the screened compounds

# The encoding of this file is heavily corrupted, so we'll just choose any old single-byte encoding to get it to load. We won't really use the columns with corruption anyway.

# In[12]:


pb_data = pd.read_csv('cmpd_info_COCA_PDX_20170629_full.csv', encoding='iso-8859-1')
broad_ids = pb_data['broad_id'].copy()


# In[13]:


pb_data = pb_data[['broad_id', 'pref_name', 'registered_name', 'smiles', 'cas_number']]
pb_data.set_index('broad_id', inplace=True, verify_integrity=True)
pb_data = pb_data.rename(columns={'pref_name': 'name', 'smiles': 'smiles_original'})
pb_data['smiles_original'].replace('0', '', inplace=True)


# ## Fix up "doubled" structures
# Many SMILES in this file consist of two identical fragments. We can reliably fix these by dropping one of the duplicates. This is done before tautomer canonicalization because a too-small value for `max_tautomers` will lead to different generated tautomers for each fragment.

# In[14]:


def fix_doubled(mol):
    frags = AllChem.GetMolFrags(mol, asMols=True)
    if len(frags) == 2:
        frag_smiles = [AllChem.MolToSmiles(f, isomericSmiles=True) for f in frags]
        if frag_smiles[0] == frag_smiles[1]:
            mol = MolFromSmiles(frag_smiles[0])
    return mol

pb_data['mol'] = pb_data['smiles_original'].map(MolFromSmiles).map(fix_doubled)


# ## Structure normalization
# Generate canonical tautomers.

# In[15]:


pb_data['mol'] = parallel_progress_map(canonicalize_tautomers, pb_data['mol'])


# ## Fix up known salts

# In[16]:


SALT_SMILES = {'[Na]', 'Cl'}
# Canonicalize smiles.
salt_smiles_canonical = {AllChem.MolToSmiles(MolFromSmiles(s), isomericSmiles=True) for s in SALT_SMILES}


# In[17]:


def fix_salts(mol):
    frags = AllChem.GetMolFrags(mol, asMols=True)    
    if len(frags) > 1:
        frag_smiles = [AllChem.MolToSmiles(f, isomericSmiles=True) for f in frags]
        keep = [s for s in frag_smiles if s not in salt_smiles_canonical]
        if keep:
            mol = MolFromSmiles('.'.join(keep))
    return mol

pb_data['mol'] = pb_data['mol'].map(fix_salts)


# ## Delete specific fragments from specific records
# These are fragments that don't belong in a general "salts" list.

# In[18]:


DELETE_FRAGMENT_SMARTS_MAP = {
    'BRD-M92352362-002-01-5': ['O=[Si]=O', 'O=[Mg]'],
    'BRD-M97113494-001-01-9': ['[Zn+2]'],
    'BRD-M71717789-001-01-1': ['C'],
    'BRD-M50817856-304-01-0': ['C'],
    'BRD-M72047285-001-01-1': ['C'],
    'BRD-M71786117-001-02-7': ['CCO'],
    'BRD-M64062803-001-01-3': ['NCCCCC(N)C(O)=O'],
}


# In[19]:


for broad_id, patterns in DELETE_FRAGMENT_SMARTS_MAP.items():
    mol = pb_data.loc[broad_id, 'mol']
    for pattern in patterns:
        pm = AllChem.MolFromSmarts(pattern)
        mol = AllChem.DeleteSubstructs(mol, pm, onlyFrags=True)
        AllChem.SanitizeMol(mol)
    pb_data.loc[broad_id, 'mol'] = mol


# ## Apply SMARTS replacements on specific records.

# In[20]:


REPLACE_SMARTS_MAP = {
    'BRD-M12718521-001-01-4': ('[$([O;D1]-c1c(-[O;D2]-[C;D1])cccc1)].[O;D1]-[C;D1]', 'OC'),
    'BRD-A34095931-001-01-9': ('[$([C;D2]-[N;D3])]=[O;D1]', 'C'),
    'BRD-K50394339-001-02-9': ('[$([N;D2]-[C;D3]=[O;D1])]-[C;D1]', 'N'),
}


# In[21]:


for broad_id, (pattern, replacement) in REPLACE_SMARTS_MAP.items():
    mol = pb_data.loc[broad_id, 'mol']
    pm = AllChem.MolFromSmarts(pattern)
    rm = AllChem.MolFromSmiles(replacement)
    mol = AllChem.ReplaceSubstructs(mol, pm, rm, replaceAll=True)[0]
    AllChem.SanitizeMol(mol)
    pb_data.loc[broad_id, 'mol'] = mol


# ## Overwrite specific structures with existing, better ones
# This mostly addresses missing or incorrect stereochemistry detected with the self-similarity = 1 review. The tuples are `(destination, source)`.

# In[22]:


OVERWRITE_IDS = [
    ('BRD-A37704979-001-07-3', 'BRD-A70514680-003-10-4'),
    ('BRD-A82396632-001-15-4', 'BRD-K34022604-001-04-1'),
    ('BRD-K74514084-001-02-1', 'BRD-A62036252-001-01-9'),
    ('BRD-A09722536-001-02-6', 'BRD-A74975734-004-08-1'),
    ('BRD-A31811760-001-04-2', 'BRD-A26503646-001-14-1'),
    ('BRD-A01098288-001-01-1', 'BRD-K69253806-001-01-2'),
]


# In[23]:


for dest, src in OVERWRITE_IDS:
    pb_data.loc[dest, 'mol'] = copy.deepcopy(pb_data.loc[src, 'mol'])


# ## Add missing structures and full structure overrides

# In[24]:


NEW_STRUCTURE_MAP = {
    # Missing
    'BRD-U14272896-000-01-9': 'COCC[n+]1c(C)n(Cc2cnccn2)c3C(=O)c4ccccc4C(=O)c13',
    'BRD-K85178109-001-01-5': 'Fc1cnc(nc1Nc2nc3N(C(=O)C(Oc3cc2)(C)C)COP(=O)(O)O)Nc4cc(OC)c(OC)c(OC)c4',
    'BRD-U21835532-000-01-4': 'CC1(C)CC(O)CC(C)(C)N1[O]',
    'BRD-U78772829-000-01-3': 'Fc1cc(c(F)cc1F)C[C@@H](N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F',
    'BRD-U68849542-000-01-7': 'Fc1cc(c(F)cc1F)C[C@@H](N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F',
    'BRD-U56662427-000-01-8': 'O=C(OCC)/C1=C/[C@@H](OC(CC)CC)[C@H](NC(=O)C)[C@@H](N)C1',
    'BRD-U32963902-000-01-3': '[Zn+2].[O-][n+]1ccccc1[S-].[O-][n+]2ccccc2[S-]',
    'BRD-K13183738-300-01-7': 'N=C(N)c1ccc(OCCCCCOc2ccc(C(=N)N)cc2)cc1',
    # Overrides
    'BRD-M80278122-318-01-7': 'N.N.O=C([O-])C1(C(=O)[O-])CCC1.[Pt+2]',
    'BRD-K69172251-001-07-1': 'N.N.[Cl-].[Cl-].[Pt+2]',
    'BRD-M78823677-034-01-6': 'N[C@@H]1CCCC[C@H]1N.O=C(O)C(=O)O.[Pt]',
    # Overrides that should be done with SMARTS replacements, but I don't have time to deal with that now!
    'BRD-K68202742-001-11-6': 'CC(=CC(C)C=CC(O)=NO)C(=O)c1ccc(N(C)C)cc1',
    'BRD-U14272896-000-01-9': 'COCCn1c2c([n+](Cc3cnccn3)c1C)C(=O)c1ccccc1C2=O',
    'BRD-K95901403-001-03-7': 'Cc1ccc(S(=O)(=O)N=c2[nH]c3ccccc3[nH]c2=Nc2cccc3nsnc23)cc1',
    'BRD-K07859598-001-10-4': 'O=C1N=c2ccc(I)cc2=C1Cc1cc(Br)c(O)c(Br)c1',
    'BRD-K99545815-074-01-1': 'CN(C1=C(C=CC=N1)CNC2=NC(=NC=C2C(F)(F)F)NC3=CC4=C(C=C3)NC(=O)C4)S(=O)(=O)C',
    'BRD-K99545815-001-04-8': 'CN(C1=C(C=CC=N1)CNC2=NC(=NC=C2C(F)(F)F)NC3=CC4=C(C=C3)NC(=O)C4)S(=O)(=O)C',
    'BRD-K83963101-001-04-4': 'C1C2=CN=C(N=C2C3=C(C=C(C=C3)Cl)C(=N1)C4=C(C=CC=C4F)F)NC5=CC=C(C=C5)C(=O)O',
}


# In[25]:


for broad_id, smiles in NEW_STRUCTURE_MAP.items():
    pb_data.loc[broad_id, 'mol'] = canonicalize_tautomers(MolFromSmiles(smiles))


# ## Correct CAS numbers, only where the format is invalid
# There's no good way of validating CAS numbers automatically, but if the string isn't even structurally a valid CAS number then it's definitely bad.

# In[26]:


CAS_MAP = {
    'BRD-A85295731-001-01-5': '1953-02-2',
    'BRD-K79116891-003-13-6': '499-67-2',
    'BRD-K37890730-001-14-4': '7689-03-4',
    'BRD-K35520305-001-15-1': '4342-03-4',
    'BRD-K75958195-037-07-8': '15574-96-6',
    'BRD-A91831168-003-01-7': '13074-31-2',
    'BRD-K49481516-004-17-5': '357-70-0',
    'BRD-K16195444-003-22-9': '1491-59-4',
    'BRD-K02017404-300-02-1': '1092499-93-8',
    'BRD-K57631554-003-05-4': '106-60-5',
    'BRD-A01145011-001-04-8': '3690-10-6',
}


# In[27]:


for broad_id, cas_number in CAS_MAP.items():
    pb_data.loc[broad_id, 'cas_number'] = cas_number


# ## Apply ad hoc fixups

# Add a missing alternate name to provide a match with RT.

# In[28]:


pb_data.loc['BRD-K77638923-001-01-2', 'registered_name'] += ', CUDC-907'


# Fix incorrect name (was XL765). The erroneous identification of this compound is well-known in the medicinal chemistry field.

# In[29]:


pb_data.loc['BRD-K75308783-001-05-4', 'name'] = 'PI3K-IN-1'


# Fix incorrect (corrupted) name. The original value contained a synonym with stray non-ASCII characters.

# In[30]:


pb_data.loc['BRD-K93460210-001-20-3', 'name'] = 'Vorapaxar'


# Normalize name in a way that our generic normalization doesn't handle (strip leading zeros after dash).

# In[31]:


pb_data.loc['BRD-K67566344-001-05-9', 'name'] = 'KU-63794'


# Copy a better structure with more fully-specified stereochemistry, then specify one more chiral center.

# In[32]:


mol = copy.deepcopy(pb_data.loc['BRD-M93222215-330-01-3', 'mol'])
rxn = AllChem.ReactionFromSmarts('[$([C;D3]-[C;D2]-[O;D1]):1]>>[C@H:1]')
mol = rxn.RunReactants([mol])[0][0]
AllChem.SanitizeMol(mol)
pb_data.loc['BRD-A53467354-001-01-4', 'mol'] = mol


# Strip stereochemistry and perform more comprehensive tautomer searching for certain complex compounds already present in RT. These structures have tricky stereochemistry, so we'll just ignore it to assist the matching and accept whatever structure is in RT.

# In[33]:


canonicalize_tautomers_deep = molvs.tautomer.TautomerCanonicalizer(max_tautomers=20000)

def clean_hard(mol):
    AllChem.RemoveStereochemistry(mol)
    mol = canonicalize_tautomers_deep(mol)
    return mol


# In[34]:


for broad_id in {'BRD-K91658406-001-02-6', 'BRD-K84937637-001-07-3', 'BRD-K08177763-001-03-4',
                 'BRD-K60866521-001-04-8', 'BRD-K68202742-001-11-6'}:
    pb_data.loc[broad_id, 'mol'] = clean_hard(pb_data.loc[broad_id, 'mol'])


# In[35]:


for lincs_id in {'HMSL10052', 'HMSL10235', 'HMSL10277', 'HMSL10204', 'HMSL10279'}:
    rt_data.loc[lincs_id, 'mol'] = clean_hard(rt_data.loc[lincs_id, 'mol'])
    rt_data.loc[lincs_id, 'smiles'] = AllChem.MolToSmiles(rt_data.loc[lincs_id, 'mol'], isomericSmiles=True)


# ## Accept legitimate fragmented compounds

# In[36]:


ACCEPT_SMARTS = {'[Pt;D0]', '[K;D0].[I;D0]', '[Bi;D0].[O;D0]', '[Gd;D0]', '[Sr;D0]', '[Co+2D1]-[CD2]#[ND1]'}
accept_queries = {AllChem.MolFromSmarts(s) for s in ACCEPT_SMARTS}


# In[37]:


def accept_known_fragmented(mol):
    if len(AllChem.GetMolFrags(mol, asMols=True)) > 1:
        if any(mol.HasSubstructMatch(q) for q in accept_queries):
            return True
    return None

pb_data['frag_ok'] = pb_data['mol'].map(accept_known_fragmented)


# Mark one specific zinc coordination complex as OK -- we don't want to accept zinc fragments in general.

# In[38]:


pb_data.loc['BRD-U32963902-000-01-3', 'frag_ok'] = True


# ## Reject molecules with multiple fragments
# Here we begin to accumulated rejected records in the DataFrame `reject`. It's critical that rejected records are also removed from `pb_data`!

# In[39]:


def partition(df, by, force_copy=True):
    """Return df split into two DataFrames, the first where by==True and the second where by==False.
    
    If force_copy=True, copies will be returned. Otherwise the decision to make copies is left up to Pandas.
    
    """
    gb = df.groupby(by)
    ret = []
    for v in True, False:
        try:
            d = gb.get_group(v)
        except KeyError:
            d = pd.DataFrame()
        if force_copy:
            d = d.copy()
        ret.append(d)
    return ret


# In[40]:


is_fragmented = (pb_data['mol'].apply(AllChem.GetMolFrags, asMols=True).apply(len) > 1)
reject1, pb_data = partition(pb_data, is_fragmented & (pb_data['frag_ok'] != True))
reject1['reason'] = "multiple fragments"


# ## Merge compound tables on Isomeric SMILES

# In[41]:


pb_data['smiles'] = pb_data['mol'].map(lambda x: AllChem.MolToSmiles(x, isomericSmiles=True))


# In[42]:


records = pd.merge(pb_data.reset_index(), rt_data[rt_data['smiles'] != ''].reset_index(),
                   on='smiles', how='left', suffixes=('', '_rt_smiles'))
records.set_index('broad_id', inplace=True, verify_integrity=True)


# ## Reject compounds with identical smiles but no matching names

# In[43]:


def norm(s):
    s = s.lower()
    s = s.replace(' ', '')
    # Strip dashes not surrounded by parentheses (i.e. don't modify "(-)" enantiomer labels).
    s = re.sub(r'(?<!\()-(?!\))', '', s)
    return s

def name_match(brd_name, rt_name, rt_synonyms):
    rt_all = [rt_name] if pd.notnull(rt_name) else []
    if isinstance(rt_synonyms, list):
        rt_all += rt_synonyms
    return any(norm(n) in norm(brd_name) for n in rt_all)


# In[44]:


smiles_match_idx = records['mol_rt_smiles'].notnull()
name_match_idx = records.apply(lambda r: name_match(r['name'] + r['registered_name'], r['name_rt_smiles'], r['alternate_names']), axis=1)
reject2, records = partition(records, smiles_match_idx & ~name_match_idx)
reject2['reason'] = "SMILES match, name mismatch"


# ## Accept compounds with identical smiles and matching name

# In[45]:


accept_match1, records = partition(records, smiles_match_idx & name_match_idx)


# ## Compute fingerprints for all structures

# In[46]:


rt_data['fingerprint'] = rt_data['mol'].map(lambda m: AllChem.GetMorganFingerprint(m, 2))
records['fingerprint'] = records['mol'].map(lambda m: AllChem.GetMorganFingerprint(m, 2))


# ## Compute between-set structure similarity

# In[47]:


similarity = np.empty((len(rt_data), len(records)))
similarity.shape


# In[48]:


for i, fp1 in enumerate(rt_data['fingerprint']):
    similarity[i] = records['fingerprint'].map(lambda fp2: DiceSimilarity(fp1, fp2) if fp1 and fp2 else 0)


# ## Merge compound tables on nearest similarity match

# In[49]:


most_similar = similarity.argmax(axis=0)
records = pd.merge(records, rt_data.iloc[most_similar].reset_index().set_index(records.index),
                  left_index=True, right_index=True, suffixes=('','_rt_sim'))
records['similarity'] = similarity.max(axis=0)


# In[50]:


plt.figure(figsize=(9,4))
plt.hist(records['similarity'], bins=30);
plt.title("Maximum similarity value distribution")
plt.tight_layout()


# In[51]:


sim_name_match_idx = records.apply(lambda r: name_match(r['name'] + r['registered_name'],
                                                        r['name_rt_sim'], r['alternate_names_rt_sim']),
                                   axis=1)
sim_1_idx = records['similarity'] == 1


# ## Accept compounds with similarity=1 match and matching name

# In[52]:


accept_match2, records = partition(records, sim_1_idx & sim_name_match_idx)


# ## Reject compounds with similarity=1 match but no matching names

# This is a list of false positives that should not be rejected.

# In[53]:


reject3_ok = {'BRD-K39188321-001-13-8'}


# In[54]:


records['reject3_ok'] = False
records.loc[reject3_ok, 'reject3_ok'] = True


# In[55]:


reject3, records = partition(records, sim_1_idx & ~sim_name_match_idx & ~records['reject3_ok'])
reject3['reason'] = "Similarity=1 match, name mismatch"


# ## Display compounds with similarity>=0.9 for review

# In[56]:


SIM_THRESHOLD = 0.9

for broad_id, r in records[(records['similarity'] >= SIM_THRESHOLD) & (records['similarity'] < 1)].iterrows():
    AllChem.Compute2DCoords(r['mol'])
    AllChem.GenerateDepictionMatching2DStructure(r['mol_rt_sim'], r['mol'], acceptFailure=True)
    display(Box([
        VBox([Image(value=Draw.MolToImage(r['mol'], size=(500,300))._repr_png_()),
              Label('{name} : {broad_id}'.format(broad_id=broad_id, **r))]),
        VBox([Image(value=Draw.MolToImage(r['mol_rt_sim'], size=(500,300))._repr_png_()),
              Label('{name_rt_sim} : lincs_id_rt_sim'.format(**r))])
    ]))
    print(r['smiles'])
    print(r['smiles_rt_sim'])
    print(r['similarity'])


# ## Compute within-set structure similarity for remaining new compounds

# In[57]:


self_similarity = np.empty((len(records), len(records)))
self_similarity.shape


# In[58]:


for i, fp1 in enumerate(records['fingerprint']):
    self_similarity[i] = records['fingerprint'].map(lambda fp2: DiceSimilarity(fp1, fp2))
    self_similarity[i, i] = 0


# ## Drop RT-related columns to simplify new compounds table

# In[59]:


columns_drop_idx = records.columns.str.contains(r'(?:_rt_|lincs_id|alternate_names|similarity)')
records_rt_data = records.loc[:, columns_drop_idx]
records = records.loc[:, ~columns_drop_idx]


# ## Merge new compounds with self on nearest similarity match

# In[60]:


most_self_similar = self_similarity.argmax(axis=0)
records = pd.merge(records, records.iloc[most_self_similar].reset_index().set_index(records.index),
                   left_index=True, right_index=True, suffixes=('','_self_sim'))
records['self_similarity'] = self_similarity.max(axis=0)
records.rename(columns={'broad_id':'broad_id_self_sim'}, inplace=True)


# In[61]:


plt.figure(figsize=(9,4))
plt.hist(records['self_similarity'], bins=30);
plt.title("Maximum self-similarity value distribution")
plt.tight_layout()


# ## Display within-group similarity=1 matches (but not identical SMILES) for review

# In[62]:


self_sim_1_idx = records['self_similarity'] == 1
smiles_different_idx = records['smiles'] < records['smiles_self_sim']

for broad_id, r in records.loc[self_sim_1_idx & smiles_different_idx].iterrows():
    AllChem.Compute2DCoords(r['mol'])
    AllChem.Compute2DCoords(r['mol_self_sim'])
    display(Box([
        VBox([Image(value=Draw.MolToImage(r['mol'], size=(500,300))._repr_png_()),
              Label('{name} : {broad_id}'.format(broad_id=broad_id, **r))]),
        VBox([Image(value=Draw.MolToImage(r['mol_self_sim'], size=(500,300))._repr_png_()),
              Label('{name_self_sim} : {broad_id_self_sim}'.format(**r))])
    ]))
    print(r['smiles'])
    print(r['smiles_self_sim'])


# ## Display records with no SMILES for review

# In[63]:


records.loc[records['smiles'] == '', ['name', 'registered_name']]


# ## Display rejected records for review

# In[64]:


reject = pd.concat([reject1, reject2, reject3], verify_integrity=True)
reject.rename(columns={'lincs_id':'lincs_id_rt_smiles'}, inplace=True)


# In[65]:


if len(reject) == 0:
    display(HTML('<h2 style="color: green">Congratulations, all records were accepted!</h2>'))
for broad_id, r in reject.iterrows():
    AllChem.Compute2DCoords(r['mol'])
    cells = [
        VBox([Label('{name} : {broad_id}'.format(broad_id=broad_id, **r)),
              Image(value=Draw.MolToImage(r['mol'], size=(500,300))._repr_png_())])
    ]
    for suffix in 'rt_smiles', 'rt_sim':
        other_mol = r['mol_' + suffix]
        if pd.notnull(other_mol):
            AllChem.GenerateDepictionMatching2DStructure(other_mol, r['mol'], acceptFailure=True)
            other_name = r['name_' + suffix]
            other_id = r['lincs_id_' + suffix]
            cells.append(
                VBox([Label('{name} : ({lincs_id})'.format(name=other_name, lincs_id=other_id)),
                     Image(value=Draw.MolToImage(other_mol, size=(500,300))._repr_png_())])
            )
            break
    display(VBox([HTML('<h3>{reason}</h3>'.format(**r)), Box(cells)]))


# In[66]:


accept_match = pd.concat([accept_match1, accept_match2]).reindex(columns=accept_match2.columns)
accept_new = records


# ## Parse primary and alternate names from single column
# The original datafile's `pref_name` column contains strings generally of the form `primary (alt1, alt2, ...)`. The following regular expression extracts the names in almost all cases, but a few oddballs do slip through. Critically, parentheses and commas in primary names like `(-)-Huperzine A`, `epiafzelechin (2r,3r)(-)`, and IUPAC nomenclature are generally handled correctly. A handful of records have comma and/or semicolon separated lists without the grouping parentheses; these are not handled and the resulting primary name will contain the full string.

# In[67]:


names = accept_new['name'].str.extractall(r'^(?P<name>.*?)(?: \((?P<alternate_names>[^(]+)\) ?)?$')
names.reset_index(level='match', drop=True, inplace=True)
names['alternate_names'] = names['alternate_names'].str.split(', ?')


# In[68]:


accept_new = pd.concat([accept_new.drop('name', axis=1), names], axis=1)


# ## Normalize missing values in several columns

# In[69]:


accept_new['cas_number'].replace(['0'], [None], inplace=True)
accept_new['alternate_names'] = accept_new['alternate_names'].map(lambda x: x if isinstance(x, list) else [])


# ## Build lists of all names from existing RT records and new records

# In[70]:


def records_to_names(df):
    alternate_names = df['alternate_names'].apply(pd.Series).stack().reset_index(level=1, drop=True)
    names = pd.concat([df.name, alternate_names])
    names = names.str.replace(r'[^A-Za-z0-9]', '').str.lower()
    names = pd.DataFrame({'name_clean': names})
    return names


# In[71]:


names_rt = records_to_names(rt_data)
names_new = records_to_names(accept_new)


# ## Display new records that match an RT record on name
# These are compounds that might have a bad structure in one database or the other, but the similarity fell under our 0.9 threshold above.

# In[72]:


name_conflicts = pd.merge(names_new.reset_index(), names_rt.reset_index())
name_conflicts = pd.merge(name_conflicts, accept_new, left_on='broad_id', right_index=True)
name_conflicts = pd.merge(name_conflicts, rt_data, left_on='lincs_id', right_index=True, suffixes=['', '_rt'])
name_conflicts = name_conflicts[['broad_id', 'lincs_id',
                                 'name', 'alternate_names', 'mol', 'smiles',
                                 'name_rt', 'alternate_names_rt', 'mol_rt', 'smiles_rt']]


# In[73]:


if len(name_conflicts) == 0:
    display(HTML('<h2 style="color: green">No name conflicts!</h2>'))
for _, r in name_conflicts.iterrows():
    mol1 = r['mol']
    mol2 = r['mol_rt']
    mcs_result = rdFMCS.FindMCS([mol1, mol2], ringMatchesRingOnly=True, matchValences=True)
    mcs = AllChem.MolFromSmarts(mcs_result.smartsString)
    AllChem.Compute2DCoords(mcs)
    AlignDepict(mol1, mcs)
    AlignDepict(mol2, mcs)
    alt = '(' + ', '.join(r['alternate_names']) + ')' if r['alternate_names'] else ''
    alt_rt = '(' + ', '.join(r['alternate_names_rt']) + ')' if r['alternate_names_rt'] else ''
    mol1_match = mol1.GetSubstructMatch(mcs)
    mol2_match = mol2.GetSubstructMatch(mcs)
    std_args = {'highlightColor': [0, 0.6, 0.6], 'kekulize': False, 'size': (500, 300)}
    display(Box([
        VBox([Image(value=Draw.MolToImage(mol1, highlightAtoms=mol1_match, **std_args)._repr_png_()),
              Label('{name} {alt} : {broad_id}'.format(alt=alt, **r))]),
        VBox([Image(value=Draw.MolToImage(mol2, highlightAtoms=mol2_match, **std_args)._repr_png_()),
              Label('{name_rt} {alt} : {lincs_id}'.format(alt=alt_rt, **r))]),
    ]))
    print(r['smiles'])
    print(r['smiles_rt'])


# 
# ## Ensure we only matched each compound to an RT compound by only one method

# In[74]:


assert all(accept_match.lincs_id.notnull() ^ accept_match.lincs_id_rt_sim.notnull()), "SMILES/similarity match conflict"


# ## Ensure all original records are accounted for

# In[75]:


final_broad_ids = functools.reduce(pd.Index.append, [d.index for d in (accept_match, accept_new, reject)])
assert set(broad_ids) == set(final_broad_ids), "Records lost or created"


# # Prepare the new records for HMSL ID assignment and JSON export

# ## Generate InChI and InChIKey columns

# In[76]:


accept_new['inchi'] = accept_new['mol'].map(AllChem.MolToInchi)
accept_new['inchi_key'] = accept_new['inchi'].map(AllChem.InchiToInchiKey)


# In[77]:


accept_new = accept_new[['name', 'alternate_names', 'smiles', 'inchi', 'inchi_key', 'cas_number']]


# ## Build list of unique structures, taking record with most identifiers
# First we consider synonym count, then presence of a CAS number.

# In[78]:


accept_new['name_priority'] = accept_new.apply(lambda r: (len(r['alternate_names']), pd.notnull(r['cas_number'])), axis=1)
g_smiles = accept_new[accept_new['smiles'] != ''].sort_values('name_priority', ascending=False).groupby('smiles')
new_compounds = g_smiles.nth(0).drop('name_priority', axis=1).reset_index()


# ## Add records with empty smiles

# In[79]:


new_compounds_no_smiles = accept_new.loc[accept_new.smiles == '', ['name', 'alternate_names', 'cas_number']]
new_compounds = new_compounds.append(new_compounds_no_smiles, ignore_index=True)


# ## Assign new HMSL IDs

# In[80]:


new_compounds['lincs_id'] = ['HMSL%5d' % i for i in range(10700, 10700 + len(new_compounds))]


# In[81]:


new_compounds['type'] = 'small_molecule'
new_compounds['curated_by'] = 'JLM'


# In[82]:


new_compounds.head(10)


# ## Write out JSON for importing into ReagentTracker

# In[83]:


new_compounds.to_json('new_small_molecules.json', orient='records')


# ## Build mapping from broad_id to lincs_id and write out as CSV

# In[147]:


id_map_match = accept_match['lincs_id'].fillna(accept_match['lincs_id_rt_sim']).reset_index()
id_map_new_1 = pd.merge(accept_new.loc[accept_new.smiles != ''].reset_index(), new_compounds, on='smiles')[['broad_id', 'lincs_id']]
id_map_new_2 = pd.merge(accept_new.loc[accept_new.smiles == ''].reset_index(), new_compounds, on='name')[['broad_id', 'lincs_id']]
id_map = pd.concat([id_map_match, id_map_new_1, id_map_new_2], ignore_index=True)
assert set(broad_ids) == set(id_map['broad_id']), "id_map has missing or extra records"
id_map.to_csv('bhola_broad_hmsl_id_mapping.csv', index=False)

