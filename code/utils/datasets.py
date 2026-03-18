import os

from sklearn.model_selection import train_test_split
from torchmil.datasets import CAMELYON16MILDataset, PANDAMILDataset, RSNAMILDataset

FOLDS_DIR = "/work/work_fjaviersaezm/benchmark-mil/train_splits/"
DATA_DIR = "/data/datasets/"
CONT_DATA_DIR = "/data/data_fjaviersaezm/"

DATASET_DIR = {
    "rsna": f"{DATA_DIR}/RSNA_ICH/MIL_processed/",
    "panda": f"{DATA_DIR}/PANDA/PANDA_original/",
    "camelyon16": f"{DATA_DIR}/CAMELYON16/",
}


def keep_only_existing_files(path, names, ext=".npy"):
    existing_files = []
    for name in names:
        file = f"{path}/{name}{ext}"
        if os.path.isfile(file):
            existing_files.append(name)
    return existing_files


def load_fold_names(folds_dir, fold):
    with open(os.path.join(folds_dir, f"train_{fold}.txt"), "rb") as f:
        train_names = f.read().decode().splitlines()
    with open(os.path.join(folds_dir, f"val_{fold}.txt"), "rb") as f:
        val_names = f.read().decode().splitlines()
    return train_names, val_names


def load_dataset(config, mode="train_val"):

    if config.dataset.name == "rsna":

        dataset_dir = DATASET_DIR["rsna"]
        features_dir = config.dataset.features_dir
        partition = "train" if mode == "train_val" else "test"

        dataset = RSNAMILDataset(
            dataset_dir,
            features_dir,
            partition=partition,
            adj_with_dist=True,
            norm_adj=True,
            load_at_init=config.load_at_init,
            bag_keys=config.bag_keys,  # Changed from config.model.bag_keys
        )

    elif config.dataset.name in ["panda", "camelyon16"]:

        dataset_dir = DATASET_DIR[config.dataset.name]

        # patches_dir = config.dataset.patches_dir
        patch_size = config.dataset.patch_size
        features_dir = config.dataset.features_dir

        partition = "train" if mode == "train_val" else "test"

        if config.dataset.name == "panda":
            dataset = PANDAMILDataset(
                dataset_dir,
                features_dir,
                partition=partition,
                patch_size=patch_size,
                adj_with_dist=True,
                norm_adj=True,
                load_at_init=config.load_at_init,
                bag_keys=config.bag_keys,  # Changed from config.model.bag_keys
            )
        elif config.dataset.name == "camelyon16":
            dataset = CAMELYON16MILDataset(
                dataset_dir,
                features_dir,
                partition=partition,
                patch_size=patch_size,
                adj_with_dist=True,
                norm_adj=True,
                load_at_init=config.load_at_init,
                bag_keys=config.bag_keys,  # Changed from config.model.bag_keys
            )

    elif "lymph" in config.dataset.name.lower() or "GTEX" in config.dataset.name:

        if "lymph" in config.dataset.name.lower():
            root_path = f"{CONT_DATA_DIR}/RadboudLymph/"
        else:
            tissue_type = config.dataset.name.split("GTEX")[-1]
            root_path = f"{CONT_DATA_DIR}/GTEX/{tissue_type}/"

        from datasets.CLAMtoTorchmil import CLAMtoTorchmil

        dataset = CLAMtoTorchmil(
            root=root_path,
            features=config.dataset.features_dir,
            bag_keys=config.bag_keys,
            patch_size=config.dataset.patch_size,
            adj_with_dist=False,
            norm_adj=False,
            load_at_init=config.load_at_init,
        )

    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported")

    if mode == "train_val":

        if config.fold is None:
            print("Using random split for train/val")

            val_prop = config.val_prop
            seed = config.seed

            bags_labels = dataset.get_bag_labels()
            len_ds = len(dataset)

            idx = list(range(len_ds))
            idx_train, idx_val = train_test_split(
                idx, test_size=val_prop, random_state=seed, stratify=bags_labels
            )

        else:
            print(f"Using fold {config.fold} for train/val")

            folds_dir = f"{FOLDS_DIR}/{config.dataset.name}/"

            train_bag_names, val_bag_names = load_fold_names(folds_dir, config.fold)
            bag_names = dataset.get_bag_names()

            idx_train = [bag_names.index(bag_name) for bag_name in train_bag_names]
            idx_val = [bag_names.index(bag_name) for bag_name in val_bag_names]

        train_dataset = dataset.subset(idx_train)
        val_dataset = dataset.subset(idx_val)

        return train_dataset, val_dataset
    elif mode == "test":
        test_dataset = dataset

        return test_dataset
