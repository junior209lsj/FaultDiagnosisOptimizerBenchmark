# from cProfile import label
import wget
import zipfile
from scipy import io
import numpy as np
import pandas as pd
import os
import patoolib


def download_cwru(root: str, sample_rate="12k") -> pd.DataFrame:
    """
    Download CWRU drive end bearning dataset.
    Author: Seongjae Lee

    Parameters
    ----------
    root: str
        Root directory where the data files are saved.
    sample rate: str
        if "12k", 12 KHz data will be downloaded, or if "48k", 48 KHz data will be downloaded.
    
    Returns
    ----------
    pd.DataFrame
        Return dataframe containing data segments of the CWRU dataset.
    """
    base_url = "https://engineering.case.edu/sites/default/files/"

    if sample_rate == "48k":
        filenames = {
            # Normal
            "97.mat": "N_000_0",
            "98.mat": "N_000_1",
            "99.mat": "N_000_2",
            "100.mat": "N_000_3",
            # Inner race fault
            "109.mat": "IR_007_0",
            "110.mat": "IR_007_1",
            "111.mat": "IR_007_2",
            "112.mat": "IR_007_3",
            "174.mat": "IR_014_0",
            "175.mat": "IR_014_1",
            "176.mat": "IR_014_2",
            "177.mat": "IR_014_3",
            "213.mat": "IR_021_0",
            "214.mat": "IR_021_1",
            "215.mat": "IR_021_2",
            "217.mat": "IR_021_3",
            # Outer race fault
            # @06
            "135.mat": "OR@06_007_0",
            "136.mat": "OR@06_007_1",
            "137.mat": "OR@06_007_2",
            "138.mat": "OR@06_007_3",
            "201.mat": "OR@06_014_0",
            "202.mat": "OR@06_014_1",
            "203.mat": "OR@06_014_2",
            "204.mat": "OR@06_014_3",
            "238.mat": "OR@06_021_0",
            "239.mat": "OR@06_021_1",
            "240.mat": "OR@06_021_2",
            "241.mat": "OR@06_021_3",
            # Ball fault
            "122.mat": "B_007_0",
            "123.mat": "B_007_1",
            "124.mat": "B_007_2",
            "125.mat": "B_007_3",
            "189.mat": "B_014_0",
            "190.mat": "B_014_1",
            "191.mat": "B_014_2",
            "192.mat": "B_014_3",
            "226.mat": "B_021_0",
            "227.mat": "B_021_1",
            "228.mat": "B_021_2",
            "229.mat": "B_021_3",
        }
    else:
        filenames = {
            # Normal
            "97.mat": "N_000_0",
            "98.mat": "N_000_1",
            "99.mat": "N_000_2",
            "100.mat": "N_000_3",
            # Inner race fault
            "105.mat": "IR_007_0",
            "106.mat": "IR_007_1",
            "107.mat": "IR_007_2",
            "108.mat": "IR_007_3",
            "169.mat": "IR_014_0",
            "170.mat": "IR_014_1",
            "171.mat": "IR_014_2",
            "172.mat": "IR_014_3",
            "209.mat": "IR_021_0",
            "210.mat": "IR_021_1",
            "211.mat": "IR_021_2",
            "212.mat": "IR_021_3",
            "3001.mat": "IR_028_0",
            "3002.mat": "IR_028_1",
            "3003.mat": "IR_028_2",
            "3004.mat": "IR_028_3",
            # Outer race fault
            # @06
            "130.mat": "OR@06_007_0",
            "131.mat": "OR@06_007_1",
            "132.mat": "OR@06_007_2",
            "133.mat": "OR@06_007_3",
            "197.mat": "OR@06_014_0",
            "198.mat": "OR@06_014_1",
            "199.mat": "OR@06_014_2",
            "200.mat": "OR@06_014_3",
            "234.mat": "OR@06_021_0",
            "235.mat": "OR@06_021_1",
            "236.mat": "OR@06_021_2",
            "237.mat": "OR@06_021_3",
            # @03
            "144.mat": "OR@03_007_0",
            "145.mat": "OR@03_007_1",
            "146.mat": "OR@03_007_2",
            "147.mat": "OR@03_007_3",
            "246.mat": "OR@03_021_0",
            "247.mat": "OR@03_021_1",
            "248.mat": "OR@03_021_2",
            "249.mat": "OR@03_021_3",
            # @12
            "156.mat": "OR@12_007_0",
            "158.mat": "OR@12_007_1",
            "159.mat": "OR@12_007_2",
            "160.mat": "OR@12_007_3",
            "258.mat": "OR@12_021_0",
            "259.mat": "OR@12_021_1",
            "260.mat": "OR@12_021_2",
            "261.mat": "OR@12_021_3",
            # Ball fault
            "118.mat": "B_007_0",
            "119.mat": "B_007_1",
            "120.mat": "B_007_2",
            "121.mat": "B_007_3",
            "185.mat": "B_014_0",
            "186.mat": "B_014_1",
            "187.mat": "B_014_2",
            "188.mat": "B_014_3",
            "222.mat": "B_021_0",
            "223.mat": "B_021_1",
            "224.mat": "B_021_2",
            "225.mat": "B_021_3",
            "3005.mat": "B_028_0",
            "3006.mat": "B_028_1",
            "3007.mat": "B_028_2",
            "3008.mat": "B_028_3",
        }

    label_map = {
        ("N", "000"): 0,
        ("B", "007"): 1,
        ("B", "014"): 2,
        ("B", "021"): 3,
        ("B", "028"): 999,
        ("IR", "007"): 4,
        ("IR", "014"): 5,
        ("IR", "021"): 6,
        ("IR", "028"): 999,
        ("OR@03", "007"): 999,
        ("OR@03", "014"): 999,
        ("OR@03", "021"): 999,
        ("OR@03", "028"): 999,
        ("OR@06", "007"): 7,
        ("OR@06", "014"): 8,
        ("OR@06", "021"): 9,
        ("OR@06", "028"): 999,
        ("OR@12", "007"): 999,
        ("OR@12", "014"): 999,
        ("OR@12", "021"): 999,
        ("OR@12", "028"): 999,
    }

    if not os.path.isdir(root):
        os.makedirs(root)

    df = {}
    df["data"] = []
    df["fault_type"] = []
    df["crack_size"] = []
    df["load"] = []
    df["label"] = []
    for key, value in filenames.items():
        filename = root + "/" + value + ".mat"
        if not os.path.isfile(filename):
            os.system(f"wget -O {filename} {base_url + key}")
            # wget.download(base_url + key, filename)
        data = io.loadmat(filename)
        body = None
        for elem in data.keys():
            if "DE" in elem:
                body = data[elem]

        if body is None:
            continue

        body = np.ravel(body, order="F")

        labels = value.split("_")
        label = label_map[(labels[0], labels[1])]
        df["fault_type"].append(labels[0])
        df["crack_size"].append(labels[1])
        df["load"].append(int(labels[2]))
        df["label"].append(label)
        df["data"].append(body)

    data_frame = pd.DataFrame(df)

    return data_frame


def download_paderborn(root: str, sample: bool = False) -> pd.DataFrame:
    """
    Download Paderborn University dataset.
    The data with bearing of real damages will be downloaded.
    Please see Table 5 of the paper "Condition Monitoring of Bearing Damage in 
    Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: 
    A Benchmark Data Set for Data-Driven Classification"
    Author: Seongjae Lee

    Parameters
    ----------
    root: str
        Root directory where the data files are saved.
    sample: bool
        If true, only first data file of each bearing were used.
    
    Returns
    ----------
    pd.DataFrame
        Return dataframe containing data segments of the Paderborn University dataset.
    """
    sampling_rate = 64
    url = "http://groups.uni-paderborn.de/kat/BearingDataCenter"
    filenames = [
        ("K001", "N"),
        ("K002", "N"),
        ("K003", "N"),
        ("K004", "N"),
        ("K005", "N"),
        ("K006", "N"),
        ("KI04", "IR"),
        ("KI14", "IR"),
        ("KI16", "IR"),
        ("KI17", "IR"),
        ("KI18", "IR"),
        ("KI21", "IR"),
        ("KA04", "OR"),
        ("KA15", "OR"),
        ("KA16", "OR"),
        ("KA22", "OR"),
        ("KA30", "OR"),
    ]

    label_map = {"N": 0, "IR": 1, "OR": 2}

    domains = [
        "N15_M07_F10",
        "N09_M07_F10",
        "N15_M01_F10",
        "N15_M07_F04",
    ]

    domain_statistics = {
        "N15_M07_F10": (1500, 0.7, 1000),
        "N09_M07_F10": (900, 0.7, 1000),
        "N15_M01_F10": (1500, 0.1, 1000),
        "N15_M07_F04": (1500, 0.7, 400),
    }

    if sample:
        sample_list = [1]
    else:
        sample_list = [x + 1 for x in range(20)]

    if not os.path.isdir(root):
        os.makedirs(root)

    df = {
        "data": [],
        "fault_type": [],
        "sampling_rate": [],
        "rotational_speed(rpm)": [],
        "load_torque(Nm)": [],
        "radial_force(N)": [],
        "label": [],
    }

    for filename in filenames:
        if not os.path.isdir(f"{root}/{filename[0]}"):
            if not os.path.isfile(f"{root}/{filename[0]}.rar"):
                os.system(f"wget -O {root}/{filename[0]}.rar {url}/{filename[0]}.rar")
            patoolib.extract_archive(
                f"{root}/{filename[0]}.rar", outdir=root, interactive=False
            )
        for domain in domains:
            for data_num in sample_list:
                data = io.loadmat(
                    f"{root}/{filename[0]}/{domain}_{filename[0]}_{data_num}.mat"
                )
                y = data[f"{domain}_{filename[0]}_{data_num}"]["Y"][0][0][0]

                for i in range(len(y)):
                    flag = y[i]["Name"]
                    if flag == "vibration_1":
                        body = y[i]["Data"].ravel()
                        break

                df["data"].append(body)
                df["fault_type"].append(filename[1])
                df["sampling_rate"].append(sampling_rate)
                df["rotational_speed(rpm)"].append(domain_statistics[domain][0])
                df["load_torque(Nm)"].append(domain_statistics[domain][1])
                df["radial_force(N)"].append(domain_statistics[domain][2])
                df["label"].append(label_map[filename[1]])

    data_frame = pd.DataFrame(df)

    return data_frame


def download_mfpt(root: str) -> pd.DataFrame:
    """
    Download the MFPT dataset.
    Author: Seongjae Lee

    Parameters
    ----------
    root: str
        Root directory where the data files are saved.
    
    Returns
    ----------
    pd.DataFrame
        Return dataframe containing data segments of the MFPT dataset.
    """
    url = "https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"
    zipname = "data.zip"
    datafolder = "MFPT Fault Data Sets"

    filenames = [
        (f"{root}/{datafolder}/1 - Three Baseline Conditions/baseline_1.mat", "N"),
        (f"{root}/{datafolder}/1 - Three Baseline Conditions/baseline_2.mat", "N"),
        (f"{root}/{datafolder}/1 - Three Baseline Conditions/baseline_3.mat", "N"),
        (
            f"{root}/{datafolder}/2 - Three Outer Race Fault Conditions/OuterRaceFault_1.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/2 - Three Outer Race Fault Conditions/OuterRaceFault_2.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/2 - Three Outer Race Fault Conditions/OuterRaceFault_3.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_1.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_2.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_3.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_4.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_5.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_6.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_7.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_1.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_2.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_3.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_4.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_5.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_6.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_7.mat",
            "IR",
        ),
    ]

    label_map = {"N": 0, "IR": 1, "OR": 2}

    if not os.path.isdir(root):
        os.makedirs(root)

    if not os.path.isdir(f"{root}/{datafolder}"):
        os.system(f"wget -O {root}/{zipname} {url}")
        with zipfile.ZipFile(f"{root}/{zipname}", "r") as f:
            f.extractall(f"{root}")
        os.remove(f"{root}/{zipname}")
    else:
        print("File is already existed, use existed file.")

    df = {}
    df["data"] = []
    df["fault_type"] = []
    df["sampling_rate"] = []
    df["load"] = []
    df["shaft_rate"] = []
    df["label"] = []

    for file in filenames:
        filename = file[0]
        fault_type = file[1]
        data = io.loadmat(filename)
        sr = data["bearing"]["sr"][0][0].ravel()[0]
        body = data["bearing"]["gs"][0][0].ravel()
        load = data["bearing"]["load"][0][0].ravel()[0]
        shaft_rate = data["bearing"]["rate"][0][0].ravel()[0]

        label = label_map[fault_type]

        df["fault_type"].append(fault_type)
        df["data"].append(body)
        df["sampling_rate"].append(sr)
        df["load"].append(load)
        df["shaft_rate"].append(shaft_rate)
        df["label"].append(label)

    data_frame = pd.DataFrame(df)

    return data_frame