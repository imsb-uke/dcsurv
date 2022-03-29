import pandas as pd
import numpy as np
import h5py
import os
import defaults


def metabric(split_train_test=False):
    """Load METABRIC dataset

    Args
    ---
    - split_train_test (bool): Return original train test split

    Remarks
    ---
    - METABRIC dataset from the Curtis et al. 2012 "The genomic and
      transcriptomic architecture of 2,000 breast tumours reveals novel
      subgroups" (https://pubmed.ncbi.nlm.nih.gov/22522925/)
    - Dataset copied from
      https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/metabric
    - Feature names from Katzman et al. 2018:
      "DeepSurv: Personalized treatment recommender system using a Cox
      proportional hazards deep neural network"
      (https://doi.org/10.1186/s12874-018-0482-1)
    """

    filepath = (
        os.path.dirname(__file__) +
        "/datasets/metabric_IHC4_clinical_train_test.h5"
    )

    if not os.path.exists(filepath):
        raise ValueError(
            "Unable to load METABRIC dataset. "
            "Not found at 'datasets/metabric_IHC4_clinical_train_test.h5'"
        )

    with h5py.File(filepath, "r") as f:
        result = pd.DataFrame()
        for split in ["train", "test"]:

            df = pd.DataFrame()
            df[defaults.DURATION_COL] = np.array(
                f[split]["t"]) * 30  # convert months to days
            df[defaults.EVENT_COL] = np.array(f[split]["e"])

            features = np.array(f[split]["x"])
            df[[f"x{i}" for i in range(features.shape[1])]] = features
            df["split"] = split

            result = result.append(df)

    metabric_dtypes = {
        defaults.EVENT_COL: "category",
        "hormone_treatment": "category",
        "radiotherapy": "category",
        "chemotherapy": "category",
        "er_positive": "category",
        "split": "str",
    }

    result = (
        result.rename(
            columns={
                "x0": "marker_MKI67",
                "x1": "marker_EGFR",
                "x2": "marker_PGR",
                "x3": "marker_ERBB2",
                "x4": "hormone_treatment",
                "x5": "radiotherapy",
                "x6": "chemotherapy",
                "x7": "er_positive",
                "x8": "age",
            }
        )
        .pipe(_generate_patient_id_index, num_digits=4, prefix="m")
        .pipe(_change_dtypes, metabric_dtypes, "float32")
    )

    if split_train_test:
        return (
            result[lambda df: df.split == "train"].drop("split", axis=1),
            result[lambda df: df.split == "test"].drop("split", axis=1)
        )
    else:
        return (
            result.drop("split", axis=1)
            .reset_index(drop=True)
            .pipe(_generate_patient_id_index, num_digits=4, prefix="m")
        )


def support(split_train_test=False):
    """Load SUPPORT dataset


    Args
    ---
    - split_train_test (bool): Return original train test split

    Remarks
    ---
    - SUPPORT dataset from the Knaus et al. 1995 "Study to Understand Prognoses
      Preferences Outcomes and Risks of Treatment"
      (https://pubmed.ncbi.nlm.nih.gov/7810938/)
    - Dataset copied from
      https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/support
    - Feature names from Katzman et al. 2018:
      "DeepSurv: Personalized treatment recommender system using a Cox
      proportional hazards deep neural network"
      (https://doi.org/10.1186/s12874-018-0482-1)
    """

    filepath = os.path.dirname(__file__) + "/datasets/support_train_test.h5"

    if not os.path.exists(filepath):
        raise ValueError(f"Unable to load SUPPORT dataset at '{filepath}'")

    with h5py.File(filepath, "r") as f:
        result = pd.DataFrame()
        for split in ["train", "test"]:

            features = np.array(f[split]["x"])

            df = pd.DataFrame()
            df[defaults.DURATION_COL] = np.array(f[split]["t"])
            df[defaults.EVENT_COL] = np.array(f[split]["e"])
            df[[f"x{i}" for i in range(features.shape[1])]] = features
            df["split"] = split

            result = result.append(df)

    support_dtypes = {
        defaults.EVENT_COL: "category",
        "sex": "category",
        "race": "category",
        "diabetes": "category",
        "dementia": "category",
        "cancer": "category",
        "split": "str",
    }

    result = (
        result.rename(
            columns={
                "x0": "age",
                "x1": "sex",
                "x2": "race",
                "x3": "no_comorbidities",
                "x4": "diabetes",
                "x5": "dementia",
                "x6": "cancer",
                "x7": "mean_arterial_blood_pressure",
                "x8": "heart_rate",
                "x9": "respiration_rate",
                "x10": "temperature",
                "x11": "no_white_blood_cells",
                "x12": "serum_sodium",
                "x13": "serum_creatinine",
            }
        )
        .pipe(_generate_patient_id_index, num_digits=4, prefix="s")
        .pipe(_change_dtypes, support_dtypes, "float32")
    )

    if split_train_test:
        return (
            result[lambda df: df.split == "train"].drop("split", axis=1),
            result[lambda df: df.split == "test"].drop("split", axis=1)
        )
    else:
        return (
            result.drop("split", axis=1)
            .reset_index(drop=True)
            .pipe(_generate_patient_id_index, num_digits=4, prefix="s")
        )


def flchain(drop_chapter=True):
    """Load FLCHAIN dataset

    Args
    ---
    - split_train_test (bool): Return original train test split

    Remarks
    ---
    - FLCHAIN dataset from Knaus et al. 1995 "Terry M. Therneau. A package for
      survival analysis" (https://CRAN.R-project.org/package=survival/)
    """
    filepath = os.path.dirname(__file__) + "/datasets/flchain.h5"

    if not os.path.exists(filepath):
        raise ValueError(f"Unable to load FLCHAIN dataset at '{filepath}'")

    result = pd.read_hdf(filepath)

    flchain_dtypes = {
        "flc_group": "category",
        "mgus": "category",
        "chapter": "category",
        defaults.EVENT_COL: "category",
    }

    result = (
        result.rename(
            columns={
                "sample.yr": "sample_yr",
                "flc.grp": "flc_group",
                "futime": defaults.DURATION_COL,
                "death": defaults.EVENT_COL,
            }
        )
        .assign(
            sex=lambda df: df["sex"].apply(
                lambda x: 1.0 if x == "F" else 0.0 if x == "M" else np.nan
            )
        )
        .pipe(_generate_patient_id_index, num_digits=4, prefix="f")
        .pipe(_change_dtypes, flchain_dtypes, "float32")
    )

    if drop_chapter:
        result = result.drop("chapter", axis=1)

    return result


def _change_dtypes(df, dtype_dict, default_dtype="float32"):
    for col in df:
        df[col] = df[col].astype(dtype_dict.get(col, default_dtype))

    return df


def _generate_patient_id_index(df, prefix="pat", num_digits=4):
    return df.assign(
        patient_id=lambda df: [
            f"{prefix}-{idx:0{num_digits}d}" for idx in df.index]
    ).set_index("patient_id")


def get_dataset(dataset_name):
    return {
        'support': support(),
        'metabric': metabric(),
        'flchain': flchain(),
    }[dataset_name]
