"""Central dataset paths. Override via environment variables; defaults assume a
local ./data directory. No paths are hard-coded to a user home directory.

  export JESTER_ROOT=/path/to/20bn-jester-v1
  export JESTER_DIR=/path/to/jester-csvs        # holds the *-train/validation/labels csv
  export IPN_ROOT=/path/to/ipn/frames
  export IPN_ANNOT_DIR=/path/to/ipn/annotations
  export ARID_ROOT=/path/to/arid/clips
  export ARID_LIST_DIR=/path/to/arid/splits
  export SSV2_DIR=/path/to/ssv2/videos
  export SSV2_CSV_DIR=/path/to/ssv2/csvs
"""
import os

# --- Jester (jpg frames) ---
JESTER_ROOT = os.environ.get("JESTER_ROOT", "data/jester/20bn-jester-v1")
_JD = os.environ.get("JESTER_DIR", "data/jester")
JESTER_LABELS = os.path.join(_JD, "jester-v1-labels.csv")
JESTER_CSV = {"train": os.path.join(_JD, "jester-v1-train.csv"),
              "val": os.path.join(_JD, "jester-v1-validation.csv")}

# --- IPN Hand (jpg frames sliced by annotation ranges) ---
IPN_ROOT = os.environ.get("IPN_ROOT", "data/ipn/frames")
_IA = os.environ.get("IPN_ANNOT_DIR", "data/ipn/annotations")
IPN_ANNOT = {"train": os.path.join(_IA, "Annot_TrainList.txt"),
             "val": os.path.join(_IA, "Annot_TestList.txt")}

# --- ARID v1.5 (mp4 clips); official split1 lists with .avi->.mp4 remap ---
ARID_ROOT = os.environ.get("ARID_ROOT", "data/arid/clips_v1.5")
_AL = os.environ.get("ARID_LIST_DIR", "data/arid")
ARID_LIST = {"train": os.path.join(_AL, "ARID_split1_train.txt"),
             "val": os.path.join(_AL, "ARID_split1_test.txt")}

# --- Something-Something v2 (webm) [optional] ---
SSV2_DIR = os.environ.get("SSV2_DIR", "data/ssv2/videos")
_SC = os.environ.get("SSV2_CSV_DIR", "data/ssv2")
SSV2_CSV = {"train": os.path.join(_SC, "train.csv"), "val": os.path.join(_SC, "val.csv")}
