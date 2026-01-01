import pydicom
import pandas as pd
import random
from pathlib import Path
import shutil

random.seed(0)

DICOM_ROOT = Path("/mnt/d/Datasets/NYU_fastMRI/fastMRI_brain_DICOM")
OUTPUT_ROOT = Path("/mnt/d/Datasets/NYU_fastMRI/brain_dicom_t1t2_dataset")
SUBJECTS_LIST_FILE = Path("./subjects_list.csv")
RELEVANT_T1_PROTOCOLS = ['AX T1', 'AX T1 BRAIN', 'AX T1 fse', 'AxT1', 'T1 AXIAL', 'T1 AXL', 'Axial T1 FSE', 'Axial T1 SE']
RELEVANT_T2_PROTOCOLS = ['AX T2', 'AX T2 BRAIN', 'T2 AXL', 'Axial T2', 'Axial T2 FSE', 'T2 TSE AXIAL']
NUM_FOLDS = 3


def make_crossval_splits(sub_list):
    
    num_subs = len(sub_list)
    random.shuffle(sub_list)
    
    split_names = {}
    for fold in range(1, NUM_FOLDS+1):
        num_test_subs = num_subs // NUM_FOLDS
        if fold == 1:
            split_list = ['test'] * num_test_subs + ['train'] * (num_subs - num_test_subs)
        elif fold == NUM_FOLDS:
            split_list = ['train'] * (num_subs - num_test_subs) + ['test'] * num_test_subs
        else:
            split_list = ['train']*round(num_subs*((fold-1)/NUM_FOLDS)) + ['test']*num_test_subs
            split_list += ['train'] * (num_subs - len(split_list))

        split_names[f'fold{fold}'] = split_list

    # Write file
    splits = {'subject': sub_list}
    splits.update(split_names)
    splits = pd.DataFrame.from_dict(splits)
    splits.to_csv(OUTPUT_ROOT / Path("crossval_splits.csv"))


def main():
    
    OUTPUT_ROOT.mkdir(exist_ok=True)    

    sub_df = pd.read_csv(SUBJECTS_LIST_FILE)
    sub_list = sub_df['subject']
    t1w_protocols, t2w_protocols = sub_df['t1'], sub_df['t2']

    for i, sub_id in enumerate(sub_list):
        
        # Record which DCM files are of which contrast
        sub_dir = DICOM_ROOT / Path(sub_id)
        dcm_paths = list(sub_dir.glob('*'))
        contrasts = {}
        for dcm_path in dcm_paths:
            dataset = pydicom.dcmread(str(dcm_path))
            contrast = dataset.SeriesDescription            
            if contrast not in contrasts.keys(): contrasts[contrast] = []
            contrasts[contrast].append(str(dcm_path))
       
        t1_key, t2_key = t1w_protocols[i], t2w_protocols[i]

        # Copy T1 and T2 DCM slices
        output_sub_dir = OUTPUT_ROOT / Path(f"dicom/{sub_id}")        
        output_t1_dir = output_sub_dir / Path("t1")
        output_t2_dir = output_sub_dir / Path("t2")
        output_t1_dir.mkdir(exist_ok=True, parents=True)
        output_t2_dir.mkdir(exist_ok=True)
        for slice_path in contrasts[t1_key]:
            shutil.copy(slice_path, output_t1_dir)
        for slice_path in contrasts[t2_key]:
            shutil.copy(slice_path, output_t2_dir)
    
    make_crossval_splits(sub_list)


if __name__ == '__main__':
    main()