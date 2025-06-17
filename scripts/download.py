import os
from roboflow import Roboflow
import json
import shutil
from dotenv import load_dotenv

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if ROBOFLOW_API_KEY is None:
    print("Error: ROBOFLOW_API_KEY not found in environment variables. Please set it in your .env file.")


def download_dataset(api_key: str, workspace: str, project: str, version: int, format: str, target_dir: str):
    """
    Downloads the dataset from Roboflow.

    Args:
        api_key (str): Roboflow API key.
        workspace (str): The name of Roboflow workspace.
        project (str): The name of Roboflow project.
        version (int): The version number of the dataset to download.
        format (str): The desired format for the downloaded dataset.
        target_dir (str): The directory where the dataset will be saved.
    """
    print(f"Attempting to download dataset '{workspace}/{project} v{version}' to '{target_dir}'...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(format, location=target_dir)
    print(f"Dataset '{workspace}/{project} v{version}' downloaded to: {dataset.location}")
    return dataset.location


def _merge_coco_split_data(ds1_ann_path, ds1_img_dir, ds2_ann_path, ds2_img_dir, merged_ann_path, merged_img_dir):
    """Merges annotation data and images from two COCO dataset splits into a single new split.

    Handles merging of categories (ensuring uniqueness), re-indexing of image and annotation IDs
    to avoid conflicts, and copying of image files. If an annotation or image directory for a
    source dataset split is not found or is empty, it will be skipped for that part of the merge.

    Args:
        ds1_ann_path (str | None): Path to the COCO annotation JSON file for the first dataset's split.
                                   Can be None if this split doesn't exist for dataset 1.
        ds1_img_dir (str | None): Path to the image directory for the first dataset's split.
                                  Can be None if this split doesn't exist for dataset 1.
        ds2_ann_path (str | None): Path to the COCO annotation JSON file for the second dataset's split.
                                   Can be None if this split doesn't exist for dataset 2.
        ds2_img_dir (str | None): Path to the image directory for the second dataset's split.
                                  Can be None if this split doesn't exist for dataset 2.
        merged_ann_path (str): Path where the merged COCO annotation JSON file will be saved.
        merged_img_dir (str): Path to the directory where images from both datasets will be copied for the merged split.

    Returns:
        None: This function writes the merged annotation file and copies images directly.
    """
    print(f"Merging split: {ds1_ann_path} and {ds2_ann_path} into {merged_ann_path}")

    if not os.path.exists(ds1_ann_path):
        print(f"Warning: Annotation file not found {ds1_ann_path}, skipping merge for this part of ds1.")
        data1 = {'images': [], 'annotations': [], 'categories': []}
    else:
        with open(ds1_ann_path, 'r') as f:
            data1 = json.load(f)

    if not os.path.exists(ds2_ann_path):
        print(f"Warning: Annotation file not found {ds2_ann_path}, skipping merge for this part of ds2.")
        data2 = {'images': [], 'annotations': [], 'categories': []}
    else:
        with open(ds2_ann_path, 'r') as f:
            data2 = json.load(f)
    
    if not data1['images'] and not data1['annotations'] and not data2['images'] and not data2['annotations']:
        print(f"Both datasets are empty for this split ({ds1_ann_path}, {ds2_ann_path}). Skipping merge.")
        return

    os.makedirs(merged_img_dir, exist_ok=True)

    merged_data = {
        'info': data1.get('info', {}),
        'licenses': data1.get('licenses', []),
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Create a single "Wildfire" category
    wildfire_category_id = 1
    merged_data['categories'].append({
        'id': wildfire_category_id,
        'name': 'Wildfire',
        'supercategory': 'none'
    })

    # Create mappings from old category IDs to the new "Wildfire" category ID
    ds1_cat_id_map = {}
    if data1.get('categories'):
        for cat in data1['categories']:
            ds1_cat_id_map[cat['id']] = wildfire_category_id

    ds2_cat_id_map = {}
    if data2.get('categories'):
        for cat in data2['categories']:
            ds2_cat_id_map[cat['id']] = wildfire_category_id

    # Process dataset 1
    max_img_id_ds1 = 0
    img_id_map_ds1 = {}
    for img in data1.get('images', []):
        old_img_id = img['id']
        new_img_id = old_img_id

        img_id_map_ds1[old_img_id] = new_img_id
        img['id'] = new_img_id
        merged_data['images'].append(img)
        if os.path.exists(ds1_img_dir):
            shutil.copy(os.path.join(ds1_img_dir, img['file_name']), os.path.join(merged_img_dir, img['file_name']))
        else:
            print(f"Warning: Image directory {ds1_img_dir} not found for dataset 1 split.")
        if new_img_id > max_img_id_ds1: # Keep track of max id from ds1
            max_img_id_ds1 = new_img_id


    max_ann_id_ds1 = 0
    for ann in data1.get('annotations', []):
        old_ann_id = ann['id']

        ann['id'] = old_ann_id
        ann['image_id'] = img_id_map_ds1[ann['image_id']]
        # Map to Wildfire category
        if ann['category_id'] in ds1_cat_id_map:
            ann['category_id'] = wildfire_category_id
        else:
            print(f"Warning: category_id {ann['category_id']} not found in ds1_cat_id_map for annotation {ann['id']}. "
                  "Attempting to assign to Wildfire category.")
            ann['category_id'] = wildfire_category_id
        merged_data['annotations'].append(ann)
        if old_ann_id > max_ann_id_ds1:
            max_ann_id_ds1 = old_ann_id


    # Process dataset 2
    img_id_map_ds2 = {}
    current_max_img_id = max_img_id_ds1
    
    for img in data2.get('images', []):
        old_img_id = img['id']
        current_max_img_id += 1
        new_img_id = current_max_img_id
        img_id_map_ds2[old_img_id] = new_img_id
        
        original_file_name = img['file_name']
        
        img['id'] = new_img_id
        merged_data['images'].append(img)

        if os.path.exists(ds2_img_dir):
            source_image_path = os.path.join(ds2_img_dir, original_file_name)
            target_image_path = os.path.join(merged_img_dir, img['file_name'])
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, target_image_path)
            else:
                print(f"Warning: Image file {source_image_path} not found for dataset 2 split.")
        else:
            print(f"Warning: Image directory {ds2_img_dir} not found for dataset 2 split.")

    current_max_ann_id = max_ann_id_ds1
    for ann in data2.get('annotations', []):
        if ann['image_id'] not in img_id_map_ds2:
            print(f"Warning: image_id {ann['image_id']} for annotation in ds2 not found in img_id_map_ds2. "
                  "Skipping annotation.")
            continue
        ann['image_id'] = img_id_map_ds2[ann['image_id']]
        # Map to Wildfire category
        if ann['category_id'] in ds2_cat_id_map:
            ann['category_id'] = wildfire_category_id
        else:
            # Similar handling as for ds1
            print(f"Warning: category_id {ann['category_id']} not found in ds2_cat_id_map for annotation in ds2. "
                  "Attempting to assign to Wildfire category.")
            ann['category_id'] = wildfire_category_id

        current_max_ann_id += 1
        ann['id'] = current_max_ann_id
        merged_data['annotations'].append(ann)

    with open(merged_ann_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
    print(f"Successfully merged split and saved to {merged_ann_path} and images to {merged_img_dir}")


def merge_datasets(
        ds1_root_path,
        ds2_root_path,
        merged_root_path,
        annotation_filename="_annotations.coco.json"):
    """
    Merges two datasets, assumed to be in COCO format and downloaded from Roboflow,
    split by split (e.g., 'train', 'valid', 'test').

    It iterates through common subdirectories (splits) found in both dataset root paths.
    For each split, it calls `_merge_coco_split_data` to combine the annotations and images.
    The merged dataset will be created at `merged_root_path`.

    Args:
        ds1_root_path (str): Root directory path of the first dataset.
        ds2_root_path (str): Root directory path of the second dataset.
        merged_root_path (str): Path where the merged dataset will be created.
        annotation_filename (str, optional): Name of the COCO annotation file within each split directory.
                                             Defaults to "_annotations.coco.json".
    Returns:
        None: This function creates the merged dataset directory structure and files.
    """
    print(f"Starting dataset merge: '{ds1_root_path}' + '{ds2_root_path}' -> '{merged_root_path}'")
    os.makedirs(merged_root_path, exist_ok=True)

    splits_ds1 = []
    if os.path.exists(ds1_root_path):
        splits_ds1 = [d for d in os.listdir(ds1_root_path) 
                     if os.path.isdir(os.path.join(ds1_root_path, d))]
    
    splits_ds2 = []
    if os.path.exists(ds2_root_path):
        splits_ds2 = [d for d in os.listdir(ds2_root_path)
                     if os.path.isdir(os.path.join(ds2_root_path, d))]
    
    common_splits = set(splits_ds1).union(set(splits_ds2))

    if not common_splits:
        print("No common splits (train, valid, test etc.) found to merge.")
        return

    for split in common_splits:
        print(f"Processing split: {split}")
        ds1_split_path = os.path.join(ds1_root_path, split)
        ds2_split_path = os.path.join(ds2_root_path, split)
        merged_split_path = os.path.join(merged_root_path, split)

        ds1_ann = os.path.join(ds1_split_path, annotation_filename)
        ds1_img = ds1_split_path
        
        ds2_ann = os.path.join(ds2_split_path, annotation_filename)
        ds2_img = ds2_split_path

        merged_ann = os.path.join(merged_split_path, annotation_filename)
        merged_img = merged_split_path

        ds1_split_exists = os.path.isdir(ds1_split_path) and os.path.exists(ds1_ann)
        ds2_split_exists = os.path.isdir(ds2_split_path) and os.path.exists(ds2_ann)

        if not ds1_split_exists and not ds2_split_exists:
            print(f"Split '{split}' not found or annotation file missing in both datasets. Skipping.")
            continue
            
        os.makedirs(merged_split_path, exist_ok=True)
        os.makedirs(merged_img, exist_ok=True)

        _merge_coco_split_data(
            ds1_ann if ds1_split_exists else None, 
            ds1_img if ds1_split_exists else None, 
            ds2_ann if ds2_split_exists else None, 
            ds2_img if ds2_split_exists else None, 
            merged_ann, 
            merged_img
        )
    print(f"--- Dataset merging finished. Merged dataset at: {merged_root_path} ---")


if __name__ == "__main__":
    base_data_dir = "data"
    os.umask(0)
    os.makedirs(base_data_dir, mode=0o777, exist_ok=True)

    dataset1_name = "fire-dataset-4dah5_v1"
    dataset2_name = "fire-detection2-wayva_v1"
    merged_dataset_name = "merged_wildfire_dataset"

    dataset1_dir = os.path.join(base_data_dir, dataset1_name)
    dataset2_dir = os.path.join(base_data_dir, dataset2_name)
    merged_dataset_dir = os.path.join(base_data_dir, merged_dataset_name)

    print("--- Starting dataset downloads ---")

    download_dataset(
        api_key=ROBOFLOW_API_KEY,
        workspace="fire-dataset-je3e9", 
        project="fire-dataset-4dah5",
        version=1,
        format="coco",
        target_dir=dataset1_dir
    )

    download_dataset(
        api_key=ROBOFLOW_API_KEY,
        workspace="pfc-dshky",
        project="fire-detection2-wayva",
        version=1,
        format="coco",
        target_dir=dataset2_dir
    )
    print("--- All datasets downloaded. Starting merge process. ---")

    merge_datasets(dataset1_dir, dataset2_dir, merged_dataset_dir)

    print("--- All datasets downloaded and merged successfully! ---")