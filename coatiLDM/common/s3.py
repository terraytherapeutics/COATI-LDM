import datetime
import os
import pickle
import pandas as pd
import pytz
from urllib.parse import urlparse

import torch
from tqdm import tqdm
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from coatiLDM.constants import FIGURE_DATA_PATH


def load_figure_file(figure_filename, local_dir, filetype="pkl", has_header=True):
    s3_path = os.path.join(FIGURE_DATA_PATH, figure_filename)
    bucket, prefix = split_s3_path(s3_path)
    sync_s3_file_to_local_dir(bucket, prefix, local_dir=local_dir)
    local_path = os.path.join(local_dir, figure_filename)
    if filetype == "pkl":
        with open(local_path, "rb") as f:
            return pickle.load(f)
    elif filetype == "csv":
        if has_header:
            return pd.read_csv(local_path)
        else:
            return pd.read_csv(local_path, header=None)
    elif filetype == "pt":
        return torch.load(local_path)
    else:
        raise ValueError(f"filetype {filetype} not supported")


def split_s3_path(s3_path):
    components = urlparse(s3_path)
    # Remove the leading '/' from the path
    prefix = components.path[1:]
    return components.netloc, prefix


def sync_s3_file_to_local_dir(bucket_name, prefix, local_dir="./", verbose=True):
    """
    Sync s3 file to local disc if s3 file modified time > local modified time (or file does not exist)
    Default dir is user's home, otherwise set via S3_CACHE_DIR env
    """
    # Initialize a session using boto3
    session = boto3.Session()

    # Use the session to create a resource
    s3 = session.resource(
        "s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED)
    )

    # Make sure the directory exists
    os.makedirs(local_dir, exist_ok=True)
    filename = prefix.split("/")[-1]
    local_file_path = os.path.join(local_dir, filename)

    # Get object summary for the file on s3
    s3_obj = s3.Object(bucket_name, prefix)

    # If local file exists, compare modification times
    if os.path.exists(local_file_path):
        # Get modification time of local file
        local_file_mtime = os.path.getmtime(local_file_path)
        local_file_dt = datetime.datetime.fromtimestamp(
            local_file_mtime,
            datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo,
        ).astimezone(pytz.utc)

        # Get 'LastModified' time of s3 object
        s3_obj_dt = s3_obj.last_modified.astimezone(pytz.utc)

        # Download file if it was modified on s3 after the local copy
        if s3_obj_dt > local_file_dt:
            if verbose:
                print(
                    f"Re-downloading {prefix} from {bucket_name}, {s3_obj_dt} > {local_file_dt}"
                )
            s3_obj.download_file(local_file_path)
            if verbose:
                print(f"File updated successfully at {local_file_path}")
        # else:
        #     print(f"File at {local_file_path} is up-to-date.")
    else:
        # If local file doesn't exist, just download
        if verbose:
            print(f"Downloading {prefix} from {bucket_name}")
        s3_obj.download_file(local_file_path)
        if verbose:
            print(f"File downloaded successfully to {local_file_path}")

    return local_file_path


def sync_s3_to_local(bucket_name, prefix, verbose=True, home_dir=None):
    """
    Sync s3 file to local disc if s3 file modified time > local modified time (or file does not exist)
    Default dir is user's home, otherwise set via S3_CACHE_DIR env
    """
    # Initialize a session using boto3
    session = boto3.Session()

    # Use the session to create a resource
    s3 = session.resource(
        "s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED)
    )

    # # Get home directory
    if home_dir is None:
        home_dir = os.getenv("S3_CACHE_DIR", os.path.expanduser("~"))

    # Generate local file path
    local_file_path = os.path.join(home_dir, prefix)
    local_file_dir = os.path.dirname(local_file_path)

    # Make sure the directory exists
    os.makedirs(local_file_dir, exist_ok=True)

    # Get object summary for the file on s3
    s3_obj = s3.Object(bucket_name, prefix)

    # If local file exists, compare modification times
    if os.path.exists(local_file_path):
        # Get modification time of local file
        local_file_mtime = os.path.getmtime(local_file_path)
        local_file_dt = datetime.datetime.fromtimestamp(
            local_file_mtime,
            datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo,
        ).astimezone(pytz.utc)

        # Get 'LastModified' time of s3 object
        s3_obj_dt = s3_obj.last_modified.astimezone(pytz.utc)

        # Download file if it was modified on s3 after the local copy
        if s3_obj_dt > local_file_dt:
            if verbose:
                print(
                    f"Re-downloading {prefix} from {bucket_name}, {s3_obj_dt} > {local_file_dt}"
                )
            s3_obj.download_file(local_file_path)
            if verbose:
                print(f"File updated successfully at {local_file_path}")
        # else:
        #     print(f"File at {local_file_path} is up-to-date.")
    else:
        # If local file doesn't exist, just download
        if verbose:
            print(f"Downloading {prefix} from {bucket_name}")
        s3_obj.download_file(local_file_path)
        if verbose:
            print(f"File downloaded successfully to {local_file_path}")

    return local_file_path


def copy_bucket_dir_from_s3(bucket_dir, dest_dir):
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket("terray-public")
    nfiles = len(list(bucket.objects.filter(Prefix=bucket_dir)))
    if nfiles < 1:
        print(list(bucket.objects.filter(Prefix=bucket_dir)))
        raise Exception(f"empty_s3 {bucket_dir}")
    else:
        print(f"copying {nfiles} files from {bucket_dir} to {dest_dir}")
    for obj in tqdm(bucket.objects.filter(Prefix=bucket_dir), total=nfiles):
        if not os.path.exists(os.path.dirname(dest_dir + obj.key)):
            os.makedirs(os.path.dirname(dest_dir + obj.key))
        bucket.download_file(obj.key, dest_dir + obj.key)  # save to same path


def download_from_s3(s3_path):
    """Simple download from s3 to local file"""

    bucket_name, prefix = split_s3_path(s3_path)
    local_file_path = sync_s3_to_local(bucket_name, prefix, verbose=True)
    return local_file_path


class cache_read:
    VALID_MODES = ["rb", "r"]
    """Given full s3_uri with bucket name, sync it locally if needed, open it"""

    def __init__(self, s3_path, mode, verbose=True):
        self.s3_path = s3_path
        if mode not in self.VALID_MODES:
            raise ValueError(f'"{mode}" not in {self.VALID_MODES}')
        self.mode = mode
        self.local_file_path = None
        self.file = None
        self.verbose = verbose

    def __enter__(self):
        if os.path.isfile(self.s3_path):
            self.local_file_path = self.s3_path
        else:
            bucket_name, prefix = split_s3_path(self.s3_path)
            self.local_file_path = sync_s3_to_local(
                bucket_name, prefix, verbose=self.verbose
            )

        if self.local_file_path is not None:
            self.file = open(self.local_file_path, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()