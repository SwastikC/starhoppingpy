import os
import glob
import subprocess
import sys
import numpy as np
from scipy.ndimage import uniform_filter

from . import zfuncs as zf

def get_file_list(data_dir):
    pattern = os.path.join(data_dir, 'SPH*.[0-9][0-9][0-9].fits.Z')
    #pattern = os.path.join(data_dir, 'SPH*.[0-9][0-9]0.fits.Z')
    files = glob.glob(pattern)
    print("files:", files)
    if not files:
        return []
    files_str = " ".join(files)
    print("files_str:", files_str)
    command = (
        f"dfits {files_str} | "
        "fitsort OBS.NAME INS4.MODE SEQ.ARM DPR.TYPE INS.COMB.ICOR "
        "INS1.FILT.NAME INS.COMB.IFLT INS2.OPTI2.NAME DET.SEQ1.DIT DET.NDIT "
        "OBS.CONTAINER.ID ORIGFILE INS2.COMB.IFS"
    )
    print("command:", command)
    result = subprocess.check_output(command, shell=True, text=True)
    result_str = result.strip()
    print("Length of result_str:", len(result_str))
    lines = result_str.split("\n")
    print("Number of lines after splitting:", len(lines))
    return lines[1:] if lines else []

def parse_dfits_line(line):
    fields = line.strip().split('\t')
    if len(fields) < 14:
        fields += [''] * (14 - len(fields))
    return {
        "fname": os.path.basename(fields[0].strip()),
        "object": fields[1].strip(),
        "ins4_mode": fields[2].strip(),
        "arm": fields[3].strip(),
        "dprtype": fields[4].strip(),
        "ins_comb_icor": fields[5].strip(),
        "ins1_filt_name": fields[6].strip(),
        "filt": fields[7].strip(),
        "prism": fields[8].strip(),
        "dit": float(fields[9].strip()) if fields[9].strip() else None,
        "ndit": int(fields[10].strip()) if fields[10].strip() else None,
        "contid": fields[11].strip(),
        "origfile": fields[12].strip(),
        "ifslamp": fields[13].strip(),
    }

def group_datasets(file_entries):
    groups = {}
    for entry in file_entries:
        # Skip non-science files: if the object contains "Calibration" or if 'arm' is empty.
        if "Calibration" in entry["object"]:
            continue
        if not entry["arm"]:
            continue

        fname = entry.get("fname", "")
        print("Debug: fname =", fname)
        if fname.startswith("SPHER."):
            parts = fname.split("SPHER.", 1)[1].split("T", 1)
            print("Debug: parts =", parts)
            night = parts[0] if parts and parts[0] else "unknown"
            print("Debug: extracted night =", night)
        else:
            night = "unknown"

        dataset_id = f"day{night}-cid{entry['contid']}-{entry['arm']}"
        groups.setdefault(dataset_id, []).append(entry)
    return groups

def process_science_files(dataset_id, science_entries, data_dir, dest_dir, msgfile):
    """
    Process science files for a given dataset:
      - Create a dataset directory under dest_dir (named by dataset_id).
      - Remove old list/info files.
      - Write a new 'scilist.txt' and 'sciinfo.txt' with science file info.
      - Create symlinks in the dataset directory to the original files in data_dir.
    """
    dataset_dir = os.path.join(dest_dir, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Remove old list/info files if they exist.
    for filename in ["scilist.txt", "sciinfo.txt", "scilist_other.txt"]:
        path = os.path.join(dataset_dir, filename)
        if os.path.exists(path):
            os.remove(path)
    
    # Write out science file list and info.
    sci_list_path = os.path.join(dataset_dir, "scilist.txt")
    sci_info_path = os.path.join(dataset_dir, "sciinfo.txt")
    with open(sci_list_path, "w") as flist, open(sci_info_path, "w") as finfo:
        header = ("# FILE\tOBJECT\tINS4.MODE\tARM\tDPR.TYPE\tINS.COMB.ICOR\t"
                  "INS1.FILT.NAME\tINS.COMB.IFLT\tINS2.OPTI2.NAME\tDET.SEQ1.DIT\tDET.NDIT\n")
        finfo.write(header)
        for entry in science_entries:
            flist.write(entry["fname"] + "\n")
            line = (f"{entry['fname']}\t{entry['object']}\t{entry['ins4_mode']}\t{entry['arm']}\t"
                    f"{entry['dprtype']}\t{entry['ins_comb_icor']}\t{entry['ins1_filt_name']}\t"
                    f"{entry['filt']}\t{entry['prism']}\t{entry['dit']}\t{entry['ndit']}\n")
            finfo.write(line)
    
    # Create symlinks in the dataset directory for each science file.
    for entry in science_entries:
        src = os.path.join(data_dir, entry["fname"])
        dst = os.path.join(dataset_dir, entry["fname"])
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    
    print(f"Processed science files for dataset {dataset_id} into directory {dataset_dir}")
    # Optionally, log the message to msgfile (here we just print)
    return dataset_dir

def process_calibration_flat(all_entries, science_entry, data_dir, dataset_dir, msgfile):
    """
    Process calibration FLAT files for a given dataset.
    
    For non-IFS science files, the original IDL logic uses the science file's
    'prism' as the matching key. It then selects candidate flat files where dprtype
    contains "FLAT" and the candidate's prism, arm, and contid (if present) match the
    science file's values.
    
    This version prints additional debug info and relaxes the contid check:
    If a candidate's contid is empty, it is accepted as matching.
    """
    from datetime import datetime
    import os

    def extract_timestamp(fname):
        # Expected format: "SPHER.YYYY-MM-DDTHH:MM:SS.sss.fits.Z"
        try:
            ts_str = fname.split("SPHER.", 1)[1].split(".fits")[0]
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except Exception as e:
            print(f"Error extracting timestamp from {fname}: {e}")
            return None

    # Skip if science file is IFS.
    if science_entry["arm"].upper() == "IFS":
        print("Science file is IFS; skipping flat calibration processing.")
        return

    var = science_entry["prism"]
    print(f"Science file prism: {var}")
    print(f"Science file arm: {science_entry['arm']}, contid: {science_entry['contid']}")

    # Debug: list all entries with 'FLAT' in dprtype.
    print("All FLAT entries in all_entries:")
    for entry in all_entries:
        if "FLAT" in entry["dprtype"].upper():
            print(f"  {entry['fname']} - dprtype: {entry['dprtype']}, prism: {entry['prism']}, arm: {entry['arm']}, contid: {entry['contid']}")

    # Filter candidate flat files.
    # Relax the container ID check: if candidate contid is empty, treat it as a match.
    candidates = [
        entry for entry in all_entries
        if "FLAT" in entry["dprtype"].upper() and
           entry["prism"] == var and
           entry["arm"] == science_entry["arm"] and
           (entry["contid"] == science_entry["contid"] or entry["contid"] == "")
    ]
    print("Filtered FLAT candidates:")
    for cand in candidates:
        print(f"  {cand['fname']} - dprtype: {cand['dprtype']}, prism: {cand['prism']}, arm: {cand['arm']}, contid: '{cand['contid']}'")

    if not candidates:
        print("No calibration FLAT candidates found. Check flatoptions file.")
        return

    science_ts = extract_timestamp(science_entry["fname"])
    for candidate in candidates:
        candidate_ts = extract_timestamp(candidate["fname"])
        if candidate_ts and science_ts:
            candidate["timediff"] = abs((candidate_ts - science_ts).total_seconds())
        else:
            candidate["timediff"] = float('inf')
        print(f"Candidate {candidate['fname']} timediff: {candidate['timediff']} seconds")

    # Select candidate with the minimum time difference.
    selected = min(candidates, key=lambda x: x["timediff"])
    mintimediff = selected["timediff"]
    print(f"Selected flat: {selected['fname']} with timediff {mintimediff} seconds")
    if mintimediff > 2 * 24 * 3600:
        print("Warning: nearest flat is more than 2 days away from science.")

    # Write out calibration flat info.
    flat_list_path = os.path.join(dataset_dir, "flatlist.txt")
    flat_info_path = os.path.join(dataset_dir, "flatinfo.txt")
    with open(flat_list_path, "w") as flist, open(flat_info_path, "w") as finfo:
        header = "# FILE\tDPR.TYPE\tTIMEDIFF (sec)\tDIT\n"
        finfo.write(header)
        flist.write(selected["fname"] + "\n")
        line = f"{selected['fname']}\t{selected['dprtype']}\t{selected['timediff']}\t{selected['dit']}\n"
        finfo.write(line)

    # Create a symlink in the dataset directory.
    src = os.path.join(data_dir, selected["fname"])
    dst = os.path.join(dataset_dir, selected["fname"])
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    print(f"Processed calibration FLAT files for dataset in {dataset_dir}.")

def process_calibration_sky(all_entries, science_entry, data_dir, dataset_dir, msgfile):
    """
    Process calibration SKY files for a given dataset.
    
    For SKY calibrations, if the science file's arm is IFS, the matching key is the 'prism'
    field; otherwise, it's the 'filt' field.
    
    This function:
      - Determines the matching key from the science entry.
      - Filters candidate SKY files where dprtype contains "SKY", the candidate's arm matches,
        and its matching key (prism or filt) equals the science entry's value.
      - Computes the time differences (using timestamps extracted from the filename).
      - Finds the minimum time difference (mintimediff) and then selects all candidates with 
        timediff less than mintimediff + 57600 seconds (16 hours).
      - Writes out skylist.txt and skyinfo.txt in the dataset directory.
      - Creates symlinks in the dataset directory for the selected candidates.
    """
    from datetime import datetime
    import os

    def extract_timestamp(fname):
        # Expected format: "SPHER.YYYY-MM-DDTHH:MM:SS.sss.fits.Z"
        try:
            ts_str = fname.split("SPHER.", 1)[1].split(".fits")[0]
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except Exception as e:
            print(f"Error extracting timestamp from {fname}: {e}")
            return None

    # Determine matching key: if arm is IFS, use 'prism'; otherwise, use 'filt'
    if science_entry["arm"].upper() == "IFS":
        var = science_entry["prism"]
    else:
        var = science_entry["filt"]
    print(f"Science file matching key for SKY: {var}")
    print(f"Science file arm: {science_entry['arm']}, contid: {science_entry['contid']}")

    # Filter candidate SKY files.
    candidates = []
    for entry in all_entries:
        if "SKY" not in entry["dprtype"].upper():
            continue
        if entry["arm"] != science_entry["arm"]:
            continue
        # Relax contid check: if candidate contid is empty, accept it.
        if not (entry["contid"] == science_entry["contid"] or entry["contid"] == ""):
            continue
        # Compare matching key.
        if science_entry["arm"].upper() == "IFS":
            candidate_key = entry.get("prism", "")
        else:
            candidate_key = entry.get("filt", "")
        if candidate_key != var:
            continue
        candidates.append(entry)
    print("Filtered SKY candidates:")
    for cand in candidates:
        print(f"  {cand['fname']} - dprtype: {cand['dprtype']}, key: {candidate_key}, arm: {cand['arm']}, contid: '{cand['contid']}'")

    if not candidates:
        print("No calibration SKY candidates found. Check skyoptions file. Continuing...")
        return

    # Compute time differences.
    science_ts = extract_timestamp(science_entry["fname"])
    for candidate in candidates:
        candidate_ts = extract_timestamp(candidate["fname"])
        if candidate_ts and science_ts:
            candidate["timediff"] = abs((candidate_ts - science_ts).total_seconds())
        else:
            candidate["timediff"] = float('inf')
        print(f"Candidate {candidate['fname']} timediff: {candidate['timediff']} seconds")

    # Determine the minimum time difference.
    mintimediff = min(candidate["timediff"] for candidate in candidates)
    print(f"Minimum time difference for SKY: {mintimediff} seconds")
    if mintimediff > 2 * 24 * 3600:
        print("Warning: nearest SKY is more than 2 days away from science.")

    # Define threshold: 16 hours (57600 seconds)
    threshold = 57600
    final_candidates = [cand for cand in candidates if cand["timediff"] < mintimediff + threshold]
    print("Final SKY candidates:")
    for cand in final_candidates:
        print(f"  {cand['fname']} timediff: {cand['timediff']} seconds")

    if not final_candidates:
        print("No appropriate SKY candidates found after time filtering. Continuing...")
        return

    # Write out SKY info.
    sky_list_path = os.path.join(dataset_dir, "skylist.txt")
    sky_info_path = os.path.join(dataset_dir, "skyinfo.txt")
    with open(sky_list_path, "w") as flist, open(sky_info_path, "w") as finfo:
        header = "# FILE\tDPR.TYPE\tTIMEDIFF (sec)\tDIT\n"
        finfo.write(header)
        for cand in final_candidates:
            flist.write(cand["fname"] + "\n")
            line = f"{cand['fname']}\t{cand['dprtype']}\t{cand['timediff']}\t{cand['dit']}\n"
            finfo.write(line)

    # Create symlinks in the dataset directory.
    for cand in final_candidates:
        src = os.path.join(data_dir, cand["fname"])
        dst = os.path.join(dataset_dir, cand["fname"])
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f"Processed calibration SKY files for dataset in {dataset_dir}.")
    
 
def process_calibration_dark(all_entries, science_entry, data_dir, dataset_dir, msgfile):
    """
    Process DARK calibration files for a given dataset.
    
    For DARK calibrations:
      - If the science file is IFS:
          var = science_entry["prism"]
          condf: candidate's prism field contains "OPEN"
          condd: candidate["dit"] equals science_entry["dit"] OR candidate["dit"] < 1.7
      - Else (non‑IFS):
          var = science_entry["filt"]
          condf: candidate["filt"] equals var
          condd: candidate["dit"] equals science_entry["dit"]
    
    Candidate DARK files are those where:
          candidate["dprtype"] contains "DARK,BACK",
          condf is true,
          candidate["arm"] equals science_entry["arm"],
          condd is true,
          and candidate["contid"] equals science_entry["contid"] (or is empty).
    
    Then, compute time differences relative to the science file and select those
    within 16 hours (57600 sec) beyond the minimum difference.
    Write out "darklist.txt" and "darkinfo.txt" in the dataset directory and create symlinks.
    """
    from datetime import datetime
    import os

    def extract_timestamp(fname):
        # Expected format: "SPHER.YYYY-MM-DDTHH:MM:SS.sss.fits.Z"
        try:
            ts_str = fname.split("SPHER.", 1)[1].split(".fits")[0]
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except Exception as e:
            print(f"Error extracting timestamp from {fname}: {e}")
            return None

    # Set matching conditions based on instrument type.
    if science_entry["arm"].upper() == "IFS":
        var = science_entry["prism"]
        def condf(candidate):
            return "OPEN" in candidate["prism"].upper()
        def condd(candidate):
            return candidate["dit"] == science_entry["dit"] or (candidate["dit"] is not None and candidate["dit"] < 1.7)
    else:
        var = science_entry["filt"]
        def condf(candidate):
            return candidate["filt"] == var
        def condd(candidate):
            return candidate["dit"] == science_entry["dit"]

    print(f"Processing DARK calibration: matching key var: {var}")
    # Filter candidate DARK files.
    candidates = [
        candidate for candidate in all_entries
        if "DARK,BACK" in candidate["dprtype"].upper()
        and condf(candidate)
        and candidate["arm"] == science_entry["arm"]
        and condd(candidate)
        and (candidate["contid"] == science_entry["contid"] or candidate["contid"] == "")
    ]
    print("Filtered DARK candidates:")
    for cand in candidates:
        key_val = cand.get("filt", cand.get("prism"))
        print(f"  {cand['fname']} - dprtype: {cand['dprtype']}, key: {key_val}, arm: {cand['arm']}, dit: {cand['dit']}")
    
    if not candidates:
        print("No appropriate DARK found. Please check darkoptions file. Continuing...")
        return
    
    science_ts = extract_timestamp(science_entry["fname"])
    for candidate in candidates:
        candidate_ts = extract_timestamp(candidate["fname"])
        if candidate_ts and science_ts:
            candidate["timediff"] = abs((candidate_ts - science_ts).total_seconds())
        else:
            candidate["timediff"] = float('inf')
        print(f"Candidate {candidate['fname']} timediff: {candidate['timediff']} seconds")
    
    mintimediff = min(candidate["timediff"] for candidate in candidates)
    print(f"Minimum time difference for DARK: {mintimediff} seconds")
    if mintimediff > 2 * 24 * 3600:
        print("Warning: nearest DARK is more than 2 days away from science.")
    
    threshold = 57600  # 16 hours in seconds.
    final_candidates = [cand for cand in candidates if cand["timediff"] < mintimediff + threshold]
    print("Final DARK candidates:")
    for cand in final_candidates:
        print(f"  {cand['fname']} timediff: {cand['timediff']} seconds")
    
    if not final_candidates:
        print("No appropriate DARK candidates found after time filtering. Continuing...")
        return
    
    dark_list_path = os.path.join(dataset_dir, "darklist.txt")
    dark_info_path = os.path.join(dataset_dir, "darkinfo.txt")
    with open(dark_list_path, "w") as flist, open(dark_info_path, "w") as finfo:
        header = "# FILE\tDPR.TYPE\tTIMEDIFF (sec)\tDIT\n"
        finfo.write(header)
        for cand in final_candidates:
            flist.write(cand["fname"] + "\n")
            line = f"{cand['fname']}\t{cand['dprtype']}\t{cand['timediff']}\t{cand['dit']}\n"
            finfo.write(line)
    
    for cand in final_candidates:
        src = os.path.join(data_dir, cand["fname"])
        dst = os.path.join(dataset_dir, cand["fname"])
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    
    print(f"Processed calibration DARK files for dataset in {dataset_dir}.")

    
def process_calibration_center(all_entries, science_entry, data_dir, dataset_dir, msgfile):
    """
    Process CENTER calibration files for a given dataset.
    
    For CENTER calibration, the matching key is determined as follows:
      - If the science file's arm is IFS, use the "prism" field.
      - Otherwise, use the "filt" field.
    
    Candidate CENTER files must satisfy:
      - Their dprtype contains "CENTER" (case-insensitive).
      - Their matching key (prism or filt) is non-empty and matches the science file’s.
      - Their DIT equals the science file’s DIT.
      - Their container ID equals the science file’s container ID (or is empty).
    
    The function computes the time difference between candidate and science files
    (using the timestamp extracted from the filename), finds the minimum time difference,
    and then selects all candidates with timediff less than mintimediff + 57600 sec (16 hours).
    
    It then writes out "centerlist.txt" and "centerinfo.txt" in the dataset directory and
    creates symlinks for the selected candidates.
    """
    from datetime import datetime
    import os

    def extract_timestamp(fname):
        # Expected format: "SPHER.YYYY-MM-DDTHH:MM:SS.sss.fits.Z"
        try:
            ts_str = fname.split("SPHER.", 1)[1].split(".fits")[0]
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except Exception as e:
            print(f"Error extracting timestamp from {fname}: {e}")
            return None

    # Determine matching key based on instrument.
    if science_entry["arm"].upper() == "IFS":
        key_field = "prism"
    else:
        key_field = "filt"
    matching_value = science_entry.get(key_field, "")
    print(f"Science file {key_field}: {matching_value}")

    # Filter candidate CENTER files.
    candidates = [
        entry for entry in all_entries
        if "CENTER" in entry["dprtype"].upper() and
           entry.get(key_field, "") != "" and
           entry.get(key_field, "") == matching_value and
           entry["dit"] == science_entry["dit"] and
           (entry["contid"] == science_entry["contid"] or entry["contid"] == "")
           and entry["arm"] == science_entry["arm"]
    ]
    print("Filtered CENTER candidates:")
    for cand in candidates:
        print(f"  {cand['fname']} - dprtype: {cand['dprtype']}, {key_field}: {cand.get(key_field, '')}, DIT: {cand['dit']}, contid: '{cand['contid']}'")
    
    if not candidates:
        print("No appropriate CENTER candidates found. Check centeroptions file. Continuing...")
        return

    science_ts = extract_timestamp(science_entry["fname"])
    for candidate in candidates:
        candidate_ts = extract_timestamp(candidate["fname"])
        if candidate_ts and science_ts:
            candidate["timediff"] = abs((candidate_ts - science_ts).total_seconds())
        else:
            candidate["timediff"] = float('inf')
        print(f"Candidate {candidate['fname']} timediff: {candidate['timediff']} seconds")
    
    # Determine the minimum time difference.
    mintimediff = min(candidate["timediff"] for candidate in candidates)
    print(f"Minimum time difference for CENTER: {mintimediff} seconds")
    if mintimediff > 2 * 24 * 3600:
        print("Warning: nearest CENTER is more than 2 days away from science.")
    
    threshold = 57600  # 16 hours in seconds.
    final_candidates = [cand for cand in candidates if cand["timediff"] < mintimediff + threshold]
    print("Final CENTER candidates:")
    for cand in final_candidates:
        print(f"  {cand['fname']} timediff: {cand['timediff']} seconds")
    
    if not final_candidates:
        print("No appropriate CENTER candidates found after time filtering. Continuing...")
        return

    # Write out CENTER info.
    center_list_path = os.path.join(dataset_dir, "centerlist.txt")
    center_info_path = os.path.join(dataset_dir, "centerinfo.txt")
    with open(center_list_path, "w") as flist, open(center_info_path, "w") as finfo:
        header = "# FILE\tDPR.TYPE\tTIMEDIFF (sec)\tDIT\n"
        finfo.write(header)
        for cand in final_candidates:
            flist.write(cand["fname"] + "\n")
            line = f"{cand['fname']}\t{cand['dprtype']}\t{cand['timediff']}\t{cand['dit']}\n"
            finfo.write(line)

    # Create symlinks for each CENTER candidate.
    for cand in final_candidates:
        src = os.path.join(data_dir, cand["fname"])
        dst = os.path.join(dataset_dir, cand["fname"])
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f"Processed calibration CENTER files for dataset in {dataset_dir}.")
    
def process_calibration_flux(all_entries, science_entry, data_dir, dataset_dir, msgfile):
    """
    Process FLUX calibration files for a given dataset.
    
    For FLUX calibration, if the science file's arm is IFS, use its "prism" field as the matching key;
    otherwise, use its "filt" field.
    
    Candidate FLUX files must satisfy:
      - Their dprtype contains "FLUX".
      - Their arm matches the science file's arm.
      - Their contid matches the science file's contid (or is empty).
      - Their matching key (prism for IFS, filt for non-IFS) is not empty.
    
    The function computes time differences (using timestamps extracted from the filename) between
    each candidate and the science file. If the minimum time difference is greater than 2 hours,
    a warning is issued. It then selects all candidates with a timediff less than (mintimediff + 4 hours),
    writes out "fluxlist.txt" and "fluxinfo.txt" in the dataset directory, and creates symlinks.
    """
    from datetime import datetime
    import os

    def extract_timestamp(fname):
        # Expected format: "SPHER.YYYY-MM-DDTHH:MM:SS.sss.fits.Z"
        try:
            ts_str = fname.split("SPHER.", 1)[1].split(".fits")[0]
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except Exception as e:
            print(f"Error extracting timestamp from {fname}: {e}")
            return None

    # Determine the matching key: for IFS use "prism", else "filt"
    if science_entry["arm"].upper() == "IFS":
        var = science_entry["prism"]
        key_field = "prism"
    else:
        var = science_entry["filt"]
        key_field = "filt"
    print(f"Science file matching key for FLUX: {var}")
    print(f"Science file arm: {science_entry['arm']}, contid: {science_entry['contid']}")

    # Filter candidate FLUX files.
    candidates = [
        entry for entry in all_entries
        if "FLUX" in entry["dprtype"].upper() and
           entry["arm"] == science_entry["arm"] and
           (entry["contid"] == science_entry["contid"] or entry["contid"] == "") and
           entry.get(key_field, "") != ""
    ]
    print("Filtered FLUX candidates:")
    for cand in candidates:
        key_val = cand.get(key_field, "")
        print(f"  {cand['fname']} - dprtype: {cand['dprtype']}, {key_field}: {key_val}, arm: {cand['arm']}, contid: '{cand['contid']}'")

    if not candidates:
        print("No FLUX images found. Please check fluxoptions file. Continuing.")
        return

    science_ts = extract_timestamp(science_entry["fname"])
    for candidate in candidates:
        candidate_ts = extract_timestamp(candidate["fname"])
        if candidate_ts and science_ts:
            candidate["timediff"] = abs((candidate_ts - science_ts).total_seconds())
        else:
            candidate["timediff"] = float('inf')
        print(f"Candidate {candidate['fname']} timediff: {candidate['timediff']} seconds")

    mintimediff = min(candidate["timediff"] for candidate in candidates)
    print(f"Minimum time difference for FLUX: {mintimediff} seconds")
    if mintimediff > 7200:  # 2 hours
        print("Warning: nearest FLUX is more than 2 hours away from science.")

    threshold = 14400  # 4 hours in seconds
    final_candidates = [cand for cand in candidates if cand["timediff"] < mintimediff + threshold]
    print("Final FLUX candidates:")
    for cand in final_candidates:
        print(f"  {cand['fname']} timediff: {cand['timediff']} seconds")

    if not final_candidates:
        print("No appropriate FLUX candidates found after time filtering. Continuing.")
        return

    flux_list_path = os.path.join(dataset_dir, "fluxlist.txt")
    flux_info_path = os.path.join(dataset_dir, "fluxinfo.txt")
    with open(flux_list_path, "w") as flist, open(flux_info_path, "w") as finfo:
        header = "# FILE\tDPR.TYPE\tTIMEDIFF (sec)\tDIT\n"
        finfo.write(header)
        for cand in final_candidates:
            flist.write(cand["fname"] + "\n")
            line = f"{cand['fname']}\t{cand['dprtype']}\t{cand['timediff']}\t{cand['dit']}\n"
            finfo.write(line)

    for cand in final_candidates:
        src = os.path.join(data_dir, cand["fname"])
        dst = os.path.join(dataset_dir, cand["fname"])
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f"Processed calibration FLUX files for dataset in {dataset_dir}.")
    
def process_calibration_ifscal(all_entries, science_entry, data_dir, dataset_dir, msgfile):
    """
    Process IFS calibration files that require paired files.
    
    This function is only executed when the science file’s arm is IFS.
    
    It uses a list of calibration name strings (e.g. "CAL_BB_2_", "CAL_NB1_1_", etc.).
    For each calibration type:
      - It removes the prefix "PRI_" from the science file's prism to form a matching substring.
      - It filters candidate calibration files where:
            • dprtype contains "FLAT"
            • arm equals science_entry["arm"]
            • ifslamp contains the concatenation of the calibration string and the modified prism.
      - If candidates are found, it computes time differences relative to the science file.
      - Then, based on the number of unique DIT values among the candidates:
            • If only one unique DIT is found, select that candidate.
            • If two unique DIT values are found, select both.
            • If more than two, select the candidate with the second-lowest DIT and the one with the highest DIT.
      - The selected candidates are appended to "ifscal_list.txt" and "ifscal_info.txt" (and candidate details
        are written to "ifscal_options.txt"), and symlinks are created in the dataset directory.
    """
    from datetime import datetime
    import os

    if science_entry["arm"].upper() != "IFS":
        print("Not an IFS science file; skipping ifscal processing.")
        return

    # Remove "PRI_" prefix from science prism, if present.
    science_prism = science_entry["prism"]
    mod_prism = science_prism[4:] if science_prism.startswith("PRI_") else science_prism

    str_calnameA = ["CAL_BB_2_", "CAL_NB1_1_", "CAL_NB2_1_", "CAL_NB3_1_", "CAL_NB4_2_"]
    ifscal_list_path = os.path.join(dataset_dir, "ifscal_list.txt")
    ifscal_info_path = os.path.join(dataset_dir, "ifscal_info.txt")
    ifscal_options_path = os.path.join(dataset_dir, "ifscal_options.txt")
    
    # Overwrite options file header.
    with open(ifscal_options_path, "w") as f:
        f.write("# ifscal_options.txt\n")
    
    def extract_timestamp(fname):
        try:
            ts_str = fname.split("SPHER.", 1)[1].split(".fits")[0]
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except Exception:
            return None

    for cal_name in str_calnameA:
        pattern = cal_name + mod_prism
        print(f"Processing IFS calibration type: {cal_name} with pattern: {pattern}")
        candidates = [
            entry for entry in all_entries
            if "FLAT" in entry["dprtype"].upper() and
               entry["arm"] == science_entry["arm"] and
               pattern in entry["ifslamp"]
        ]
        print(f"Candidates for {cal_name}:")
        for cand in candidates:
            print(f"  {cand['fname']} - ifslamp: {cand['ifslamp']}")
        if not candidates:
            print(f"No {cal_name} FLAT images found. Please check {ifscal_options_path} file. Continuing...")
            continue

        # Compute time differences relative to science file.
        science_ts = extract_timestamp(science_entry["fname"])
        for candidate in candidates:
            candidate_ts = extract_timestamp(candidate["fname"])
            if candidate_ts and science_ts:
                candidate["timediff"] = abs((candidate_ts - science_ts).total_seconds())
            else:
                candidate["timediff"] = float('inf')
        # Sort candidates by DIT (lowest first)
        sorted_candidates = sorted(candidates, key=lambda x: x["dit"] if x["dit"] is not None else float('inf'))
        unique_dits = []
        for cand in sorted_candidates:
            if cand["dit"] not in unique_dits:
                unique_dits.append(cand["dit"])
        numcals = len(unique_dits)
        if numcals == 0:
            print(f"No appropriate {cal_name} FLAT images found. Please check {ifscal_options_path} file. Continuing...")
            continue
        elif numcals == 1:
            selected_candidates = [sorted_candidates[0]]
            print(f"Only one {cal_name} candidate found: {sorted_candidates[0]['fname']}")
        elif numcals == 2:
            selected_candidates = [sorted_candidates[0], sorted_candidates[-1]]
            print(f"Two {cal_name} candidates found: {sorted_candidates[0]['fname']} and {sorted_candidates[-1]['fname']}")
        else:
            selected_candidates = [sorted_candidates[1], sorted_candidates[-1]]
            print(f"More than two {cal_name} candidates found, selecting: {sorted_candidates[1]['fname']} and {sorted_candidates[-1]['fname']}")
        
        # Append calibration info.
        with open(ifscal_info_path, "a") as finfo, open(ifscal_list_path, "a") as flist:
            header = f"# Calibration: {cal_name}\n"
            finfo.write(header)
            for cand in selected_candidates:
                finfo.write(f"{cand['fname']}\t{cand['dprtype']}\t{cand.get('timediff','NA')}\t{cand['dit']}\n")
                flist.write(cand["fname"] + "\n")
                src = os.path.join(data_dir, cand["fname"])
                dst = os.path.join(dataset_dir, cand["fname"])
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
        with open(ifscal_options_path, "a") as fopt:
            fopt.write(f"# {cal_name} candidates:\n")
            for cand in candidates:
                fopt.write(f"{cand['fname']}\n")

def process_calibration_ifscal2(all_entries, science_entry, data_dir, dataset_dir, msgfile):
    """
    Process additional IFS calibration files that require only a single file.
    
    This function is executed only for IFS science files. It uses a list of calibration
    dprtype strings (e.g. "SPECPOS,LAMP", "WAVE,LAMP", "FLAT,LAMP"). For each calibration type:
      - It filters candidate files where:
            • dprtype contains the calibration type,
            • arm equals science_entry["arm"],
            • prism equals science_entry["prism"],
            • and ifslamp contains "OBS".
      - It computes the time differences relative to the science file and selects the candidate
        with the minimum time difference.
      - The selected candidate is appended to "ifscal2_list.txt" and "ifscal2_info.txt" (and candidate
        details are written to "ifscal2_options.txt"), and a symlink is created.
    """
    from datetime import datetime
    import os

    if science_entry["arm"].upper() != "IFS":
        print("Not an IFS science file; skipping ifscal2 processing.")
        return

    str_dprtype = ["SPECPOS,LAMP", "WAVE,LAMP", "FLAT,LAMP"]
    ifscal2_list_path = os.path.join(dataset_dir, "ifscal2_list.txt")
    ifscal2_info_path = os.path.join(dataset_dir, "ifscal2_info.txt")
    ifscal2_options_path = os.path.join(dataset_dir, "ifscal2_options.txt")
    with open(ifscal2_options_path, "w") as f:
        f.write("# ifscal2_options.txt\n")
    
    def extract_timestamp(fname):
        try:
            ts_str = fname.split("SPHER.", 1)[1].split(".fits")[0]
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except Exception:
            return None

    for cal_type in str_dprtype:
        print(f"Processing IFS calibration (single) type: {cal_type}")
        candidates = [
            entry for entry in all_entries
            if cal_type in entry["dprtype"].upper() and
               entry["arm"] == science_entry["arm"] and
               entry["prism"] == science_entry["prism"] and
               "OBS" in entry["ifslamp"].upper()
        ]
        print(f"Candidates for {cal_type}:")
        for cand in candidates:
            print(f"  {cand['fname']} - ifslamp: {cand['ifslamp']}")
        if not candidates:
            print(f"No {cal_type} CAL candidates found. Check ifscal2_options file. Continuing.")
            continue

        science_ts = extract_timestamp(science_entry["fname"])
        for candidate in candidates:
            candidate_ts = extract_timestamp(candidate["fname"])
            if candidate_ts and science_ts:
                candidate["timediff"] = abs((candidate_ts - science_ts).total_seconds())
            else:
                candidate["timediff"] = float('inf')
            print(f"Candidate {candidate['fname']} timediff: {candidate['timediff']} seconds")
        selected = min(candidates, key=lambda x: x["timediff"])
        print(f"Selected {cal_type} candidate: {selected['fname']} with timediff {selected['timediff']}")
        with open(ifscal2_info_path, "a") as finfo, open(ifscal2_list_path, "a") as flist:
            header = f"# Calibration: {cal_type}\n"
            finfo.write(header)
            finfo.write(f"{selected['fname']}\t{selected['dprtype']}\t{selected['timediff']}\t{selected['dit']}\n")
            flist.write(selected["fname"] + "\n")
        with open(ifscal2_options_path, "a") as fopt:
            fopt.write(f"# {cal_type} candidates:\n")
            for cand in candidates:
                fopt.write(f"{cand['fname']}\n")
        src = os.path.join(data_dir, selected["fname"])
        dst = os.path.join(dataset_dir, selected["fname"])
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

def print_science_summary(dest_dir):
    import subprocess
    import os
    # Construct the command to count lines in all "scilist.txt" files under dest_dir.
    cmd = f"wc -l {os.path.join(dest_dir, '*', 'scilist.txt')}"
    result = subprocess.check_output(cmd, shell=True, text=True)
    print("Number of science files found per concatenation:")
    print(result)
        
def irdis_makeflat(filetxt):
    # Create an IRDIS flat field from one or two FITS files listed in filetxt.
    # This function mimics the original IDL logic while automatically handling
    # files compressed with .Z. It also assumes that if an image has three dimensions,
    # the first dimension is the image index (as in IDL) and takes the median along axis 0.
    
    # Parameters:
    #  filetxt : text file containing one filename per line.
    
    # The function writes out 'flat_irdis.fits' and 'badpix_irdis.fits'.

    from .basic import fixpix 
    from astropy.io import fits    

    # Read filenames (one per line) from filetxt.
    filenames = np.loadtxt(filetxt, dtype=str)
    if filenames.ndim == 0:
        filenames = np.array([filenames])
    num = len(filenames)
    print("Number of filenames read:", num, flush=True)
    
    # Single file case.
    if num == 1:
        imgflat1, header = zf.readfits(filenames[0], header=True)
        print("Single file branch: image shape =", imgflat1.shape, flush=True)
        if imgflat1.ndim == 3:
            # In our case, axis=0 corresponds to the image index.
            imgflat1 = np.median(imgflat1, axis=0)
            print("Collapsed 3D image using median along axis 0.", flush=True)
        flat = imgflat1.copy()
        # Print some statistics from the single file.
        print("Single file flat: min =", np.min(flat), "max =", np.max(flat), flush=True)
        wnl = (np.array([]), np.array([]))
        # For the single file branch, assume no pixels were flagged.
        badpix = np.zeros_like(flat)
        print("Single file branch: Set badpix to all zeros (all pixels good).", flush=True)
    else:
        imgflat1, header1 = zf.readfits(filenames[0], header=True)
        imgflat2, header2 = zf.readfits(filenames[1], header=True)
        print("Two file branch:", flush=True)
        print("imgflat1 shape:", imgflat1.shape, "imgflat2 shape:", imgflat2.shape, flush=True)
        if imgflat1.ndim == 3:
            imgflat1 = np.median(imgflat1, axis=0)
            print("Collapsed imgflat1 along axis 0.", flush=True)
        if imgflat2.ndim == 3:
            imgflat2 = np.median(imgflat2, axis=0)
            print("Collapsed imgflat2 along axis 0.", flush=True)
        # Extract DIT values from the headers.
        dit1 = float(header1.get('HIERARCH ESO DET SEQ1 DIT', 10))
        dit2 = float(header2.get('HIERARCH ESO DET SEQ1 DIT', 4))
        print("dit1:", dit1, "dit2:", dit2, flush=True)
        ditrat = dit1 / dit2
        print("ditrat:", ditrat, flush=True)
        # Compute the image ratio.
        imgrat = imgflat1 / imgflat2
        print("imgrat: min =", np.min(imgrat), "max =", np.max(imgrat), flush=True)
        # Find pixels where the ratio deviates by more than 0.1.
        wnl = np.where(np.abs(1 - imgrat/ditrat) > 0.1)
        print("Number of flagged bad pixels (wnl):", wnl[0].size, flush=True)
        flat = imgflat1.copy()
        # Create the bad-pixel map: initialize to zero then mark flagged pixels with 1.
        badpix = flat * 0
        if wnl[0].size > 0:
            badpix[wnl] = 1
        print("Two-file branch: badpix summary: min =", np.min(badpix), "max =", np.max(badpix),
              "unique =", np.unique(badpix), flush=True)
        # Write the badpix map to a file for inspection.
        fits.writeto("debug_badpix.fits", badpix, header1, overwrite=True, output_verify='fix')
        print("Saved badpix debug file as 'debug_badpix.fits'", flush=True)
    
    # Create median image "med" from top and bottom halves.
    med = np.zeros_like(badpix)
    
    
    #med[0:1024, :] = np.median(flat[0:1024, :])
    #med[1024:, :] = np.median(flat[1024:, :])
    #print("Computed med: top half =", np.median(flat[0:1024, :]),
    #      "bottom half =", np.median(flat[1024:, :]), flush=True)
    #flat = flat / med
    fits.writeto('flat_irdis00.fits', flat, header if num==1 else header1, overwrite=True, output_verify='fix')

    print('Dealing with NaNs...')
    top_med = np.nanmedian(flat[:1024, :])
    bottom_med = np.nanmedian(flat[1024:, :])
    top_med = top_med if not np.isnan(top_med) else 1.0
    bottom_med = bottom_med if not np.isnan(bottom_med) else 1.0
    med = np.empty_like(flat)
    med[:1024, :] = top_med
    med[1024:, :] = bottom_med
    print("Computed med: top =", top_med, "bottom =", bottom_med, flush=True)
    flat = flat / med
    fits.writeto('flat_irdis01.fits', flat, header if num==1 else header1, overwrite=True, output_verify='fix')

    
    #if wnl[0].size > 0:
    #    flat[wnl] = 1

    fits.writeto('badpix_irdis00.fits', badpix, header if num==1 else header1, overwrite=True, output_verify='fix')

    # Set boundary regions to 1.
    # Set boundaries according to IDL coordinates (swapping the order):
    flat[:, 0:51] = 1       # IDL: flat[0:50,*] = 1   (columns 0-50)
    flat[0:51, :] = 1       # IDL: flat[*,0:50] = 1   (rows 0-50)
    flat[:, 920:1081] = 1    # IDL: flat[920:1080,*] = 1 (columns 920-1080)
    flat[940:, :] = 1       # IDL: flat[*,940:*] = 1   (rows 940 to end)
    flat[:, 1950:] = 1      # IDL: flat[1950:*,*] = 1  (columns 1950 to end)

    badpix[:, 0:51] = 0
    badpix[0:51, :] = 0
    badpix[:, 920:1081] = 0
    badpix[940:, :] = 0
    badpix[:, 1950:] = 0

    fits.writeto('flat_irdis02.fits', flat, header if num==1 else header1, overwrite=True, output_verify='fix')
    fits.writeto('badpix_irdis01.fits', badpix, header if num==1 else header1, overwrite=True, output_verify='fix')

    # Fix bad pixels using a 3x3 median filter on flat.
    print('bad/good pix ratio:', np.where(badpix == 1)[0].size / np.where(badpix == 0)[0].size)
    print('bad pixels:', np.where(badpix == 1)[0].size)
    print('good pixels:', np.where(badpix == 0)[0].size)
    good = 1 - badpix
    print("fixing bad pixels....", flush=True)
    flatg = fixpix(flat, good, npix=8, weight=False, noise=False, sigma=None, dc=0, silent=False)
    flatg = flatg.astype(np.float32)
    flat = np.clip(flatg, 0.8, 1.2)
    wbad = np.where((flatg < 0.8) | (flatg > 1.2))
    badpix[wbad] = 1

    # Write out the final flat and badpix maps.
    print('writing files...')
    fits.writeto('flat_irdis.fits', flat, header if num==1 else header1, overwrite=True, output_verify='fix')
    fits.writeto('badpix_irdis.fits', badpix, header if num==1 else header1, overwrite=True, output_verify='fix')
    print("Saved final flat_irdis.fits and badpix_irdis.fits", flush=True)


def irdis_make_sky(filelist, show=False):
    """
    Create a combined SKY image from a list of compressed FITS (.fits.Z) files.
    
    Parameters:
      filelist : A text file containing one FITS filename per line.
      show     : If True, display each individual sky frame and the median image.
    
    This function:
      - Reads filenames from the file.
      - Loads each file using zf.readfits (which handles .fits.Z).
      - Stacks the images into a cube and squeezes out any singleton dimension.
      - Computes the pixel-by-pixel median image.
      - Writes out the median image to "sky_irdis.fits", bypassing header verification.
    """
    import os
    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt

    # Read filenames from filelist.
    with open(filelist, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    print("Files to process:", files, flush=True)
    
    imgs = []
    headers = []
    for fname in files:
        print(f"Loading {fname}...", flush=True)
        img, hdr = zf.readfits(fname, header=True)
        imgs.append(img)
        headers.append(hdr)
    
    # Stack images into a cube (assumes each image is 2D).
    cube = np.stack(imgs, axis=0)
    print("Initial cube shape:", cube.shape, flush=True)
    
    # Squeeze extra singleton dimension if present.
    if cube.ndim == 4 and cube.shape[1] == 1:
        cube = np.squeeze(cube, axis=1)
        print("Squeezed cube shape:", cube.shape, flush=True)
    
    # Compute the median image (across the first axis).
    median_img = np.median(cube, axis=0)
    
    # Bypass header verification by ignoring non-standard header cards.
    hdr0 = headers[0]
    fits.writeto('sky_irdis.fits', median_img, hdr0, overwrite=True, output_verify='ignore')
    print("Saved combined sky image as 'sky_irdis.fits'", flush=True)
    
    if show:
        for i, img in enumerate(imgs):
            plt.imshow(img, cmap='gray')
            plt.title(f"Sky frame {i}")
            plt.show()
        plt.imshow(median_img, cmap='gray')
        plt.title("Median Sky")
        plt.show()

    
#Currently fixing too pixel 'good' pixels along with bads. Need to make less aggressive.
#But the real badpix map from above seems to be working
def irdis_badpix_from_obj2(filetxt, iter=None):
    """
    Process science images to produce a bad-pixel mask and a cleaned science frame.
    
    This function mimics the IDL procedure irdis_badpix_from_obj2:
      - Reads the flat field from 'flat_irdis.fits'.
      - If the file (filetxt) listing science FITS filenames does not exist, it exits.
      - If iter is not provided, defaults to 3.
      - Reads the science filenames from filetxt (one per line).
      - Loads the science cube from the first file using zf.readfits,
        and squeezes it so that if it has shape (n, 1, rows, cols) it becomes (n, rows, cols).
      - Uses the first image (im00) from the cube (or the cube itself if 2D) and normalizes it by the flat.
      - Applies a 10×10 median filter to im00 to form im0s, computes its standard deviation,
        and creates a mask (mimicking IDL's boolean where True becomes -1) to compute im0 = im00/mask.
      - Subtracts a 3×3 median-filtered version from im0.
      - Copies im0 into temp (setting non-finite values to 0) and writes temp as 'temp.fits'.
      - Then, for i from 0 to 5, computes a “bad” mask (where abs(temp) > 3.5×robust_sigma(temp))
        and refines temp by calling fixpix with npix = (3+i)^2.
      - Computes a final badpix map as 1 - (abs(im0 - temp) != 0), writes it to 'badpix_irdis.fits'.
      - Finally, applies fixpix on im00 using the computed badpix mask to produce imout,
        and writes that to 'sci_bp_clean_example.fits'.
    
    Note: This version bypasses header verification errors by using output_verify='ignore'
    when writing FITS files.
    """
    import os
    import numpy as np
    from astropy.io import fits
    from scipy.ndimage import median_filter
    from .basic import fixpix 

    # Helper: robust sigma (using median absolute deviation)
    def robust_sigma(arr):
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        return 1.4826 * mad

    # Read flat field (assumed uncompressed) from 'flat_irdis.fits'
    flat = fits.getdata('flat_irdis.fits')
    
    # Exit if science file list does not exist.
    if not os.path.exists(filetxt):
        print(f"File {filetxt} not found; exiting irdis_badpix_from_obj2.")
        return
    
    if iter is None:
        iter = 3
    
    # Read science filenames from filetxt.
    filenames = np.loadtxt(filetxt, dtype=str)
    if filenames.ndim == 0:
        filenames = np.array([filenames])
    
    # Load the science cube from the first file using zf.readfits.
    imc, fhead = zf.readfits(filenames[0], header=True)
    imc = np.squeeze(imc)  # Remove any singleton dimensions.
    if imc.ndim == 2:
        nimg = 1
    else:
        nimg = imc.shape[0]
    
    # Use the first image as the science frame.
    im00 = imc[0, :, :] if nimg > 1 else imc
    im00 = im00 / flat

    # Apply a 10x10 median filter to im00.
    im0s = median_filter(im00, size=10)
    std_im0s = np.std(im0s)
    # Create mask: where abs(im0s) > std_im0s, set mask = -1; else, 1.
    mask = np.where(np.abs(im0s) > std_im0s, -1, 1)
    im0 = im00 / mask
    im0 = im0 - median_filter(im0, size=3)
    
    temp = im0.copy()
    temp = temp * np.isfinite(temp)  # Set non-finite values to 0.
    
    print("Saving 'temp.fits'...", flush=True)
    fits.writeto('temp.fits', im0, fhead, overwrite=True, output_verify='ignore')
    
    # Iteratively refine temp.
    for i in range(6):
        bad = np.abs(temp) > robust_sigma(temp) * 3.5
        # fixpix expects a mask with 0 for bad pixels; convert boolean mask accordingly.
        temp = fixpix(temp, 1 - bad.astype(np.float32), npix=(3 + i)**2,
                      weight=False, noise=False, sigma=None, dc=0, silent=False)
        print(f"Iteration {i}: temp min = {np.min(temp)}, max = {np.max(temp)}", flush=True)
    
    # Compute final badpix map.
    diff = np.abs(im0 - temp)
    badpix = np.where(diff != 0, 1, 0)
    final_badpix = 1 - badpix
    print("Final badpix map: min =", np.min(final_badpix), "max =", np.max(final_badpix), flush=True)
    
    # Write final badpix map, bypassing header verification.
    fits.writeto('badpix_irdis.fits', final_badpix, fhead, overwrite=True, output_verify='ignore')
    
    # Clean the science image using fixpix on im00 with the computed badpix mask.
    imout = fixpix(im00, 1 - badpix.astype(np.float32), npix=8, weight=False, noise=False, sigma=None, dc=0, silent=False)
    fits.writeto('sci_bp_clean_example.fits', imout, fhead, overwrite=True, output_verify='ignore')
    print("Saved sci_bp_clean_example.fits", flush=True)




def qzap(img, skyfiltsize=15, boxsize=5, nsigma=4, fluxratio=0.15, 
         maxiter=2, nofluxcompare=False, nrings=0, silent=False, save=True):
    """
    Remove cosmic rays from a 2D image.
    
    Parameters
    ----------
    img : 2D numpy array
        Input image.
    skyfiltsize : int, optional
        Box size for computing local sky value (default: 15).
    boxsize : int, optional
        Box size for computing local median (default: 5).
    nsigma : float, optional
        Rejection threshold in sigma (default: 4).
    fluxratio : float, optional
        Flux comparison threshold for identifying cosmics (default: 0.15).
    maxiter : int, optional
        Number of zapping iterations (default: 2).
    nofluxcompare : bool, optional
        If True, disables the flux comparison algorithm (default: False).
    nrings : int, optional
        Radius of cosmic ray neighbors to also zap (default: 0).
    silent : bool, optional
        If True, suppresses print messages (default: False).
    save : bool, optional
        If True, saves the output image as 'qzap_out.fits' (default: True).

    Returns
    -------
    outimg : 2D numpy array
        Cleaned image with cosmic rays removed.
    outmask : 2D numpy array
        Binary mask indicating zapped pixels (1 = zapped, 0 = unchanged).
    """

    from scipy.ndimage import median_filter, gaussian_filter
    from astropy.io import fits
    from .basic import robust_sigma

    BADVAL = -1e6
    outimg = np.array(img, dtype=float)  # Ensure copy, avoid modifying original
    outmask = np.zeros_like(outimg, dtype=np.uint8)

    # Compute image sigma
    valid_pixels = outimg[outimg != BADVAL]
    sigval = np.std(valid_pixels) if valid_pixels.size > 0 else 1.0
    if not silent:
        print(f"Computed sigma: {sigval:.3f}")

    # Compute sky image
    skyimage = median_filter(outimg, size=skyfiltsize) if skyfiltsize > 0 else np.median(outimg)

    # Iterative cosmic ray removal
    nzap = 0
    for iter in range(maxiter):
        skysubimage = outimg - skyimage
        fmedimage = median_filter(skysubimage, size=boxsize)
        epsilon = np.finfo(float).eps  # Smallest positive float value
        crimage = skysubimage - fmedimage + epsilon

        peaksimage = crimage > nsigma * sigval

        if not nofluxcompare:
            peaksimage &= (fmedimage / crimage) < fluxratio

        if nrings > 0:
            peaksimage = uniform_filter(peaksimage.astype(float), size=1 + 2 * nrings) > 1e-6

        nbad = np.count_nonzero(peaksimage)
        if nbad == 0:
            break  # No more cosmic rays detected, exit loop

        outimg[peaksimage] = skyimage[peaksimage] + fmedimage[peaksimage]
        outmask[peaksimage] = 1
        nzap += nbad

        if not silent:
            print(f"Iteration {iter+1}: Number of pixels zapped = {nbad}")

    if save:
        fits.writeto('qzap_out.fits', outimg, overwrite=True)
        if not silent:
            print("Saved cleaned image as 'qzap_out.fits'")

    return outimg



def msglog(text):
    """Log a message (here, simply print)."""
    print(text, flush=True)


def sph_cen_irdis(filetxt, darkfiletxt=None, fitchans=None, badpix_file=None, lmaskoutw0=30, lmaskinw1=100, rmaskoutw0=30, rmaskinw1=100):
    """
    Find the star-center in IRDIS OBJECT,CENTER files automatically.
    
    This function mimics the IDL procedure sph_cen_irdis:
      - It uses a file (filetxt) containing the FITS filenames of the OBJECT,CENTER files.
      - It forces filetxt to 'centerlist.txt' and badpix_file to 'badpix_irdis.fits'.
      - If darkfiletxt is not provided, it checks for the existence of 'skylist.txt'
        (or 'darklist.txt') and sets darkfiletxt accordingly.
      - For each center file, it:
          * Loads the FITS file (using zf.readfits for compressed files).
          * If the image cube is 3D, it median-combines along the third dimension.
          * If darkfiletxt is provided, loads the first dark file (median-combined if 3D),
            writes it to 'dark_irdis.fits', and subtracts it from the science image.
          * If badpix_file exists, reads it, builds a bad-pixel mask bpmap = 1 - (badpix data),
            and calls fixpix on the science image.
          * Calls qzap and getfreq (dummy functions here) on the image.
          * Extracts the star-center parameters (xc, yc) from the header via esopar.
          * Splits the image into two parts (rows 0:1024 and 1024:), and calls snmap on each,
            with slightly shifted centers.
          * Blocks out unlikely regions using getrgn (which returns a mask) to zero out those pixels.
          * Then, for each half (left and right), it loops four times to find four spots:
              - Finds the maximum in the image,
              - Uses where2xy and bscentrd to get a centroid (x,y),
              - Replaces the region around the spot (via getrgn) with zeros,
              - Displays the modified image using zdisp.
          * Averages the four spot coordinates to get a center (xc1,yc1) for the left, and similarly (xc2,yc2) for the right.
          * Logs the centers using msglog.
          * If the computed centers differ too much from the expected values (with tolerance 20),
            flags the file as bad.
          * Otherwise, stores the centers in arrays.
      - After processing all files, it computes the median of the stored center coordinates
        and saves them to a file ('centers.save') using numpy's savez.
    
    Parameters:
      filetxt     : Text file with one FITS filename per line for OBJECT,CENTER files.
      darkfiletxt : (Optional) Text file with FITS filenames for dark frames.
      fitchans    : (Optional) (Not used in this translation; originally channel info.)
      badpix_file : (Optional) Filename of a bad-pixel map; if not provided, defaults to 'badpix_irdis.fits'.
      lmaskoutw0/rmaskoutw0 : outer radius of mask w0 of left/right half image(default value=30)
      lmaskinw1/rmaskinw1 : inner radius of mask w1 of left/right half image(default value=100)
    """

    from .basic import fixpix, esopar, snmap, get_region, bscentrd
    from astropy.io import fits
    from . import zfuncs as zf

    # Force filetxt and badpix_file to defaults.
    filetxt = 'centerlist.txt'
    if badpix_file is None:
        badpix_file = 'badpix_irdis.fits'
    
    # Set darkfiletxt if not provided.
    if darkfiletxt is None:
        if os.path.exists('skylist.txt'):
            darkfiletxt = './skylist.txt'
        elif os.path.exists('darklist.txt'):
            darkfiletxt = './darklist.txt'
        else:
            darkfiletxt = ''
    
    # Read the column of filenames from filetxt.
    filenames = np.loadtxt(filetxt, dtype=str)
    if filenames.ndim == 0:
        filenames = np.array([filenames])
    num = len(filenames)
    
    # Initialize arrays to hold center coordinates.
    xcs1 = np.full(num, -1.0, dtype=float)
    xcs2 = np.full(num, -1.0, dtype=float)
    ycs1 = np.full(num, -1.0, dtype=float)
    ycs2 = np.full(num, -1.0, dtype=float)
    
    # Loop through each center file.
    for ii in range(num):
        flagbad = 0
        print(f"Processing file {ii+1}/{num}: {filenames[ii]}", flush=True)
        img, head = zf.readfits(filenames[ii], header=True)
        print(np.shape(img))
        # If the image is 3D, take the median along the third dimension.
        if img.ndim == 3:
            img = np.median(img, axis=0)
        print(np.shape(img))
        # If darkfiletxt is provided, load the first dark file and subtract.
        if darkfiletxt != '':
            dark_filenames = np.loadtxt(darkfiletxt, dtype=str)
            if dark_filenames.ndim == 0:
                dark_filenames = np.array([dark_filenames])
            dark = zf.readfits(dark_filenames[0], header=True)[0]
            if dark.ndim == 3:
                dark = np.median(dark, axis=0)
            fits.writeto('dark_irdis.fits', dark, head, overwrite=True, output_verify='ignore')
            img = img - dark
            print('afterdarks:')
            print(np.shape(dark))
            print(np.shape(img))
            
        # If a badpix file exists, load it, build a mask, and run fixpix.
        if os.path.exists(badpix_file):
            bpdata = fits.getdata(badpix_file)
            bpmap = 1 - bpdata  # In IDL, badpix=1 indicates a good pixel.
            img = fixpix(img, bpmap, npix=8, weight=False, noise=False, sigma=None, dc=0, silent=False)
            print('after fixpix:')
            print(np.shape(img))
        
        # Apply qzap and getfreq (dummy functions).
        img = qzap(img)
        print('after qzap:')
        print(np.shape(img))
        img = zf.getfreq(img, 4)
        
        # Extract expected star center from header.
        #xc = float(esopar(head, 'SEQ CORO XC'))
        #yc = float(esopar(head, 'SEQ CORO YC'))
        xc = float(esopar(head, 'HIERARCH ESO SEQ CORO XC'))
        yc = float(esopar(head, 'HIERARCH ESO SEQ CORO YC'))
                       
        # Split image into two halves.
        temp = img[:,0:1024]
        print(np.shape(temp))
        stats = {
            "min": np.min(temp),
            "max": np.max(temp),
            "mean": np.mean(temp),
            "std": np.std(temp),
            "median": np.median(temp),
        }
        print(stats)
        zf.zdisp(temp)
    
        im1 = snmap(img[:,0:1024], xc=xc, yc=yc)
        im2 = snmap(img[:,1024:], xc=xc+3, yc=yc-14)
        
        # Block out unlikely regions in im1. #maskin0, maskin1
        w0 = get_region(im1, 0, lmaskoutw0, xc=xc, yc=yc)
        w1 = get_region(im1, lmaskinw1, int(512*1.6), xc=xc, yc=yc)
        im1[w0] = 0
        im1[w1] = 0
        
        # Block out unlikely regions in im2.
        w0 = get_region(im2, 0, rmaskoutw0, xc=xc+3, yc=yc-14)
        w1 = get_region(im2, rmaskinw1, int(512*1.6), xc=xc+3, yc=yc-14)
        im2[w0] = 0
        im2[w1] = 0
        
        # Initialize arrays for 4 spots.
        xcs = np.full(4, 0.0)
        ycs = np.full(4, 0.0)
        
        # Process left image (im1)
        sz = im1.shape[1]  # number of columns
        for i in range(4):
            w = np.argmax(im1)  # Returns 1D index
            sz = im1.shape[1]  # Number of columns

            # Convert 1D index to (x, y) coordinates
            x1, y1, *_ = zf.where2xy(w, sz)
            x1 = float(x1 if np.isscalar(x1) else x1[0])
            y1 = float(y1 if np.isscalar(y1) else y1[0])

            print(im1.shape)
            print('sz:',sz)
            #print('maxval:',im1[y1,x1])
            zf.writefits('temp.fits',im1)
            print('x1:', x1)
            print('y1:', y1)            

            x1b, y1b = bscentrd(im1, x1, y1)
            print('x1:', x1b)
            print('y1:', y1b)            
            x1b = float(x1b if np.isscalar(x1b) else x1b[0])
            y1b = float(y1b if np.isscalar(y1b) else y1b[0])
            # If difference > 1, update x,y.
            if abs(x1 - x1b) > 1 or abs(y1 - y1b) > 1:
                x = x1b
                y = y1b
            else:
                x = x1
                y = y1

            #import sys
            #sys.exit("Stopping the program!")  # Exit with a message

            xcs[i] = x
            ycs[i] = y
            # Remove the region around the found spot.
            wdel = get_region(im1, 0, 10, xc=x, yc=y)
            im1[wdel] = 0
            zf.zdisp(im1)
        xc1 = np.mean(xcs)
        yc1 = np.mean(ycs)
        msglog(f"left center: {xc1}, {yc1}")
        if abs(xc - xc1) > 20 or abs(yc - yc1) > 20:
            flagbad = 1
        
        # Process right image (im2)
        sz = im2.shape[1]
        for i in range(4):
            w = np.argmax(im2)  # Returns 1D index
            sz = im2.shape[1]  # Number of columns

            # Convert 1D index to (x, y) coordinates
            x1, y1, *_ = zf.where2xy(w, sz)
            x1 = float(x1 if np.isscalar(x1) else x1[0])
            y1 = float(y1 if np.isscalar(y1) else y1[0])

            print(im2.shape)
            print('sz:',sz)
            #print('maxval:',im2[y1,x1])
            zf.writefits('temp.fits',im2)
            print('x1:', x1)
            print('y1:', y1)            

            x1b, y1b = bscentrd(im2, x1, y1)
            print('x1:', x1b)
            print('y1:', y1b)            
            x1b = float(x1b if np.isscalar(x1b) else x1b[0])
            y1b = float(y1b if np.isscalar(y1b) else y1b[0])
            # If difference > 1, update x,y.
            if abs(x1 - x1b) > 1 or abs(y1 - y1b) > 1:
                x = x1b
                y = y1b
            else:
                x = x1
                y = y1
            xcs[i] = x
            ycs[i] = y
            wdel = get_region(im2, 0, 10, xc=x, yc=y)
            im2[wdel] = 0
            zf.zdisp(im2)

        xc2 = np.mean(xcs)
        yc2 = np.mean(ycs)
        msglog(f"right center: {xc2}, {yc2}")
        
        # Check centering validity.
        if flagbad or abs((xc+3) - xc2) > 20 or abs((yc-14) - yc2) > 20:
            msglog("Centering algo failed. Check Manually.")
            zf.zdisp(im2)
        else:
            # Store centers for this file.
            if ii == 0:
                xcs1 = np.full(num, -1.0)
                ycs1 = np.full(num, -1.0)
                xcs2 = np.full(num, -1.0)
                ycs2 = np.full(num, -1.0)
            xcs1[ii] = xc1
            ycs1[ii] = yc1
            xcs2[ii] = xc2
            ycs2[ii] = yc2

    # Compute final centers as the median of each list.
    xc1_final = np.median(xcs1)
    yc1_final = np.median(ycs1)
    xc2_final = np.median(xcs2)
    yc2_final = np.median(ycs2)
    
    # Save the centers to a file.
    np.savez('centers.save.npz', xc1=xc1_final, yc1=yc1_final, xc2=xc2_final, yc2=yc2_final,
             xcs1=xcs1, ycs1=ycs1, xcs2=xcs2, ycs2=ycs2, im1=im1, im2=im2)
    msglog("Saved centers to 'centers.save.npz'")
    



def irdis_make_flux_cube(fluxlist, szn=None):
    """
    Create a flux-calibrated image cube from a list of FITS files.

    Parameters
    ----------
    fluxlist : str
        Filename of a text file containing the list of FITS images.
    szn : int, optional
        Size of the cropped image. If not provided, defaults to full size.

    Returns
    -------
    fluximc : ndarray
        Final flux-calibrated image cube.
    peaks : list
        List of peak flux values.
    normflux : float
        Normalization factor for flux calibration.
    """

    from . import zfuncs as zf
    import numpy as np
    import os
    from astropy.io import fits
    from scipy import ndimage as nd
    from .basic import fixpix, bscentrd

    filenames = np.loadtxt(fluxlist, dtype=str, ndmin=1)

    # Read FITS files into a stacked image cube
    for i, fname in enumerate(filenames):
        imci, head = zf.readfits(fname, header=True)
        if i == 0:
            imc = imci
        else:
            imc = np.stack((imc, imci), axis=0)

    # Compute median if 3D
    img = np.median(imc, axis=0) if imc.ndim == 3 else imc

    sz = img.shape[1]  # Width of the image
    cc = (sz - 1) / 2.0  # Image center
    pp0 = int(cc - szn / 2.0) if szn is not None else 0

    if szn is None:
        szn = sz

    # Normalize using ND filter, exposure time, and filter width
    normflux, dit, ndfac, flambdas = irdis_normfluxfac(fluxlist, outsave="normfluxfac.save")

    # Zero out unwanted regions
    img[:, :50] = 0
    img[:, 940:1080] = 0
    img[:, 1960:] = 0

    # Apply flat field correction
    flat = zf.readfits('flat_irdis.fits')
    img /= flat

    # Handle bad pixels
    badpix = 1 - zf.readfits('badpix_irdis.fits') if os.path.exists('badpix_irdis.fits') else np.zeros_like(flat)
    fixpix(img, badpix)

    # Replace NaNs with zero
    img[np.isnan(img)] = 0

    # Split into left and right images
    img1 = img[:, :1024]
    img2 = img[:, 1024:2048]

    # ---- Process img1 ----
    temp = img1.copy()
    temp[zf.getrgn(temp, 200, 800)] = 0
    y1, x1 = np.unravel_index(np.argmax(temp), temp.shape)  # (y, x) order
    x1, y1 = bscentrd(img1, x1, y1)  # Keep (x, y) convention
    peak1 = zf.interp2d(img1, x1, y1)  # Cubic interpolation

    # Mask out flux region
    wipe = zf.getrgn(img1, 10, 800, xc=x1, yc=y1)
    fluximg1 = img1.copy()
    fluximg1[wipe] = 0

    # ---- Process img2 ----
    temp = img2.copy()
    temp[zf.getrgn(temp, 200, 800)] = 0
    y2, x2 = np.unravel_index(np.argmax(temp), temp.shape)  # (y, x) order
    x2, y2 = bscentrd(img2, x2, y2)
    peak2 = zf.interp2d(img2, x2, y2)

    wipe = zf.getrgn(img2, 10, 800, xc=x2, yc=y2)
    fluximg2 = img2.copy()
    fluximg2[wipe] = 0

    # Shift images
    dy1, dx1 = cc - y1, cc - x1  # Ensure order is (y, x)
    dy2, dx2 = cc - y2, cc - x2
    print(f"fluximg1 shape: {fluximg1.shape}, fluximg2 shape: {fluximg2.shape}")
    fluximg1 = nd.shift(fluximg1, (dx1, dy1), order=3)
    fluximg2 = nd.shift(fluximg2, (dx2, dy2), order=3)

    # Crop final images
    #pp0 = int(pp0)
    szn = int(szn)
    print(f"fluximg1 shape: {fluximg1.shape}, fluximg2 shape: {fluximg2.shape}")
    fluximg1 = fluximg1[pp0:pp0 + szn, pp0:pp0 + szn]
    fluximg2 = fluximg2[pp0:pp0 + szn, pp0:pp0 + szn]

    print(f"fluximg1 shape: {fluximg1.shape}, fluximg2 shape: {fluximg2.shape}")
    fluximc = np.stack((fluximg1, fluximg2), axis=0)
    peaks = [peak1, peak2]

    # Save results
    np.savez("flux.save.npz", peaks=peaks, fluximc=fluximc, flambdas=flambdas, normflux=normflux)

    return fluximc, peaks, normflux


def irdis_normfluxfac(filelist, outsave="normscifac.save", filename=None, fullres=False):
    """
    Compute the normalization factor for IRDIS flux calibration.

    Parameters
    ----------
    filelist : str
        Filename of a text file containing a list of FITS files.
    outsave : str, optional
        Output filename for saving results.
    filename : str, optional
        Specific FITS file to use (overrides filelist if provided).
    fullres : bool, optional
        If True, return all computed values in a named tuple. Default is False.

    Returns
    -------
    normfac : ndarray
        Normalization factor for left and right detector sides.
        If fullres=True, returns a named tuple `res` with additional results.
    """
    import numpy as np
    from . import zfuncs as zf
    from .basic import esopar
    from collections import namedtuple

    # Load filenames
    filenames = np.loadtxt(filelist, dtype=str, ndmin=1)
    
    # Read first file header
    img, head = zf.readfits(filenames[0], header=True)

    dit = float(esopar(head, 'HIERARCH ESO DET SEQ1 DIT'))
    ndfilter = esopar(head, 'HIERARCH ESO INS4 FILT2 NAME')
    filter1 = esopar(head, 'HIERARCH ESO INS1 FILT ID')
    filter2 = esopar(head, 'HIERARCH ESO INS1 OPTI2 ID')

    print(f"Using first file: {filenames[0]}")
    print(f"DIT: {dit}, Filters: {filter1}, {filter2}, {ndfilter}")

    # Lookup neutral density filter factors
    nd_factors = {
        ('0.0', 'FILT_DBF_NDH23', 'FILT_BBF_H'): 3.5,
        ('1.0', 'FILT_DBF_NDH23', 'FILT_BBF_H'): 24.9,
        ('2.0', 'FILT_DBF_NDH23', 'FILT_BBF_H'): 194.4,
        ('3.5', 'FILT_DBF_NDH23', 'FILT_BBF_H'): 3307.5,
        ('1.0', 'FILT_DBF_K12', 'FILT_BBF_Ks'): 6.7,
        ('2.0', 'FILT_DBF_K12', 'FILT_BBF_Ks'): 45.0,
        ('3.5', 'FILT_DBF_K12', 'FILT_BBF_Ks'): 690.7,
    }

    ndfac = nd_factors.get((ndfilter, filter2, filter1), 1.0)

    # Lookup wavelength ranges
    filter_wavelengths = {
        ('FILT_CLEAR', 'FILT_BBF_H'): (1.625, 0.290),
        ('FILT_DBF_H23', 'FILT_BBF_H'): (1.593, 0.052),
        ('FILT_DBF_H34', 'FILT_BBF_H'): (1.667, 0.054),
        ('FILT_CLEAR', 'FILT_BBF_Y'): (1.043, 0.140),
        ('FILT_DBF_Y23', 'FILT_BBF_Y'): (1.022, 0.049),
    }

    lambdas, dlambdas = filter_wavelengths.get((filter2, filter1), (1.0, 1.0))
    normfac = np.array([ndfac / dlambdas / dit, ndfac / dlambdas / dit])
    
    # Save results
    np.savez(outsave, normfac=normfac, dit=[dit, dit], ndfac=[ndfac, ndfac], lambdas=[lambdas, lambdas])

    # Return full results if requested
    # Define output structure
    res = namedtuple("res", ["normfac", "dit", "ndfac", "lambdas"])

    if fullres:
        return res(normfac, [dit, dit], [ndfac, ndfac], [lambdas, lambdas])

    return normfac



def irdis_basicreduc(scilist, maskfile=None, tag=''):
    """
    Perform basic reduction on IRDIS science images.

    Parameters
    ----------
    scilist : str
        Filename of a text file containing the list of science images.
    maskfile : str, optional
        Filename of the bad pixel mask file.
    tag : str, optional
        Tag to append to the output filenames.

    Returns
    -------
    out1 : str
        Filename of the reduced science image list.
    out2 : str
        Filename of the reduced reference image list.
    """

    import numpy as np
    import os
    from astropy.io import fits
    from scipy.ndimage import median_filter, uniform_filter
    from .zfuncs import readfits, writefits
    from .basic import fixpix, esopar

    darkb = readfits('dark_irdis.fits')
    sky = readfits('sky_irdis.fits')
    flat = readfits('flat_irdis.fits')
    badpix = readfits('badpix_irdis.fits')
    noise = np.random.randn(1024, 2048)

##expects all images in 2D, so added a quick diagnostic. if data is 3D, median combined.
    print("darkb shape:", darkb.shape)
    print("sky   shape:", sky.shape)
    print("flat  shape:", flat.shape)
    print("badpix shape:", badpix.shape)

    if sky.ndim == 3 and sky.shape[0] > 1:
        sky = np.median(sky, axis=0)
        print("Median-combined sky cube")

    # Load mask file if provided
    if maskfile is not None:
        mask_data = np.load(maskfile, allow_pickle=True).item()
        wmask = mask_data['wmask']
        darkb[wmask] = 0
        sky[wmask] = 0
        flat[wmask] = 1
        badpix[wmask] = 0

    # Read filenames
    filenames = np.loadtxt(scilist, dtype=str, ndmin=1)
    numf = len(filenames)
    processed_filenames = []
    ras, decs = [], []

    fcnt=0
    for fname in filenames:
        imc, head = readfits(fname, header=True)
        print(f"processing file {fcnt} of {numf}.")
        fcnt=fcnt+1
        ra = float(esopar(head, 'RA'))  # Use esopar to extract RA
        dec = float(esopar(head, 'DEC'))  # Use esopar to extract DEC
        dprtype = esopar(head,'HIERARCH ESO DPR TYPE') 
        if dprtype != 'OBJECT':
            continue
        
        ras.append(ra)
        decs.append(dec)

        # Ensure imc is always 3D
        if imc.ndim == 2:
            imc = imc[np.newaxis, :, :]  # Convert to (1, height, width)

        imgfd = imc.copy()
        num = int(head['NAXIS3']) if 'NAXIS3' in head else 1

        # Process each frame
        for i in range(num):
            #temp = (imc[i, :, :] - darkb) / flat
            temp = (imc[i, :, :] - sky) / flat
            if maskfile is not None:
                if i == 0:
                    noisefac = np.std(temp)
                temp[wmask] = noise[wmask] * noisefac
            imgfd[i, :, :] = temp

        # Bad pixel map using smoothed image
        totc = np.sum(imgfd, axis=0) if imgfd.ndim == 3 else imgfd

        def bpmask(img, size=3, thresh=15):
            imgs = median_filter(img, size=size)
            temp = img-imgs
            bmap = (np.abs(temp) > thresh * np.std(temp)).astype(np.uint8)
            return bmap

        bp1 = bpmask(totc,3, 5)
        bp2 = bpmask(totc,13,15)+0.01
        bp2 = uniform_filter(bp2, size=5) > 0.1
        bmap = bp1 * (1-bp2)
        bmap[np.isnan(bmap)] = 1  # Replace NaNs with 1
        badpix = badpix.astype(np.uint8)
        badpixfin = (bmap | badpix).astype(np.uint8)  # Bitwise OR for binary masks

        # Mask unwanted regions
        badpixfin[:, :50] = 0
        badpixfin[:, 940:1080] = 0
        badpixfin[:, 1960:] = 0

        # Apply bad pixel correction
        # Expand badpixfin along the first axis if imgfd is 3D
        if imgfd.ndim == 3 and badpixfin.ndim == 2:
            badpixfin = np.expand_dims(badpixfin, axis=0)  # Now shape (1, H, W)
            badpixfin = np.repeat(badpixfin, imgfd.shape[0], axis=0)  # Match number of frames

        imgfdb = fixpix(imgfd, 1-badpixfin)
        imgfdb[np.isnan(imgfdb)] = 0
        imgfdb[:, :50] = 0
        imgfdb[:, 940:1080] = 0
        imgfdb[:, 1960:] = 0

        # Save output
        fnamenew = 'basic' + os.path.basename(fname).replace('.Z', '')
        processed_filenames.append(fnamenew)
        writefits(fnamenew, imgfdb, head)

    # Save file list
    np.savetxt(f'basicredlist{tag}.txt', processed_filenames, fmt='%s')

    # Extract RA and DEC from FITS headers using esopar
    #ras, decs = [], []
    #for fname in processed_filenames:
    #    img, hdr = readfits(fname, header=True)  # Read image and header
    #    ra = float(esopar(hdr, 'RA'))  # Use esopar to extract RA
    #    dec = float(esopar(hdr, 'DEC'))  # Use esopar to extract DEC
    #    ras.append(ra)
    #    decs.append(dec)
    
    # Compute distances
    dras = np.abs(np.array(ras) - ras[0])
    ddecs = np.abs(np.array(decs) - decs[0])
    ds = np.sqrt(dras**2 + ddecs**2)

    # Split into two lists based on distance
    w1 = np.where(ds < 0.001)[0]
    w2 = np.where(ds >= 0.001)[0]

    out1 = f'basicredlist1{tag}.txt'
    out2 = f'basicredlist2{tag}.txt'

    if len(w1) > len(w2):
        sw = w1
        rw = w2
    else:
        sw = w2
        rw = w1
        
    if len(sw) > 0:
        np.savetxt(out1, np.array(processed_filenames)[sw], fmt='%s')
    if len(rw) > 0:
        np.savetxt(out2, np.array(processed_filenames)[rw], fmt='%s')

    return out1, out2



def irdis_make_asdi_cube(basicredlist, szn=None, normfac=None, cenbyfit=None, outfile=None, cenobid=None):
    """
    Create an ASDI cube from basic reduced images (IDL-compatible center usage).

    - Uses MEDIAN centers (xc1,yc1,xc2,yc2) from centers.save.npz for ALL science frames,
      exactly like the IDL pipeline (arrays xcs* are treated as diagnostics only).
    """
    import numpy as np
    import os
    from scipy.ndimage import shift
    from .zfuncs import readfits, gethdrfits, writefits, zdisp  # zdisp only for optional debug
    from .basic import bscentrd, esopar

    # --- 1) Normalization factors (unchanged) ---
    normfac = irdis_normfluxfac(basicredlist, outsave="normscifac.save")

    # --- 2) Load MEDIAN centers like IDL (use scalars), else fall back to getcen mode ---
    getcen = 0
    xc1_med = yc1_med = xc2_med = yc2_med = None
    if os.path.exists("centers.save.npz"):
        cen = np.load("centers.save.npz", allow_pickle=True)
        # Prefer stored medians; otherwise compute from arrays if present
        def _get_scalar(name, arrname):
            if name in cen: return float(cen[name])
            if arrname in cen and np.asarray(cen[arrname]).size > 0:
                return float(np.nanmedian(np.asarray(cen[arrname])))
            return None
        xc1_med = _get_scalar("xc1", "xcs1")
        yc1_med = _get_scalar("yc1", "ycs1")
        xc2_med = _get_scalar("xc2", "xcs2")
        yc2_med = _get_scalar("yc2", "ycs2")
        if None in (xc1_med, yc1_med, xc2_med, yc2_med):
            getcen = 1  # centers file exists but incomplete
    else:
        getcen = 1

    # --- 3) Read file list & image geometry ---
    filenames = np.loadtxt(basicredlist, dtype=str, ndmin=1)
    numf = len(filenames)

    img0, head0 = readfits(filenames[0], header=True)
    # Ensure 3D for uniform slicing
    if img0.ndim == 2:
        img0 = img0[np.newaxis, :, :]

    sz = img0.shape[2] // 2  # width of one half
    cc = (sz - 1) / 2.0
    if szn is None:
        szn = sz
        pp0 = 0
    else:
        pp0 = int(round(cc - szn / 2.0))
    ccn = (szn - 1) / 2.0

    # --- 4) Output accumulators ---
    imc, detside, pas, dits, stokes, polfilts, lambdas, tstamps = [], [], [], [], [], [], [], []

    # --- 5) Loop over science files ---modified_nav
    for j, fname in enumerate(filenames):
        img, head = readfits(fname, header=True)
        print(f"DEBUG file={fname} img.ndim={img.ndim} img.shape={img.shape} "
            f"numc_header={esopar(head,'NAXIS3')}",flush=True)

        # --- Ensure 3D ---
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        # --- Number of frames: TRUST THE DATA ---
        numc_img = img.shape[0]

        # --- Header value (diagnostic only) ---
        naxis3_hdr = (
            esopar(head, "HIERARCH ESO DET NDIT")
            or esopar(head, "NAXIS3"))

        try:
            numc_hdr = int(naxis3_hdr)
        except Exception:
            numc_hdr = None

        if numc_hdr not in (None, numc_img):
            print(f"WARNING: {fname}: header NAXIS3={numc_hdr}, "
        f"but data has {numc_img} frame(s). Using data.")

        
        numc = numc_img


        # Times / PA
        itime = esopar(head, "HIERARCH ESO DET FRAM UTC") or esopar(head, "DET FRAM UTC")
        ftime = esopar(head, "HIERARCH ESO DET SEQ UTC")  or esopar(head, "DET SEQ UTC")
        pa1   = float(esopar(head, "HIERARCH ESO TEL PARANG START") or esopar(head, "TEL PARANG START"))
        pa2   = float(esopar(head, "HIERARCH ESO TEL PARANG END")   or esopar(head, "TEL PARANG END"))

        # Dithers (detector coordinates)
        posx  = float(esopar(head, "HIERARCH ESO INS1 DITH POSX")  or esopar(head, "INS1 DITH POSX"))
        posy  = float(esopar(head, "HIERARCH ESO INS1 DITH POSY")  or esopar(head, "INS1 DITH POSY"))

        # Exposure
        dit   = float(esopar(head, "HIERARCH ESO DET SEQ1 DIT")    or esopar(head, "DET SEQ1 DIT"))

        # Filters / polar info
        filter1 = esopar(head, "HIERARCH ESO INS1 FILT ID")  or esopar(head, "INS1 FILT ID")
        filter2 = esopar(head, "HIERARCH ESO INS1 OPTI2 ID") or esopar(head, "INS1 OPTI2 ID")
        stoke12 = esopar(head, "HIERARCH ESO OCS DPI H2RT STOKES") or ""

        # --- 5a) Wavelengths: match the IDL `case` exactly ---
        if ("FILT_CLEAR" in str(filter2)) and ("FILT_BBF_H"  in str(filter1)): lambda1, lambda2 = 1.625, 1.625
        elif ("FILT_DBF_H23" in str(filter2)) and ("FILT_BBF_H"  in str(filter1)): lambda1, lambda2 = 1.593, 1.667
        elif ("FILT_DBF_H34" in str(filter2)) and ("FILT_BBF_H"  in str(filter1)): lambda1, lambda2 = 1.667, 1.733
        elif ("FILT_DBF_Y23" in str(filter2)) and ("FILT_BBF_Y"  in str(filter1)): lambda1, lambda2 = 1.022, 1.076
        elif ("FILT_DBF_J23" in str(filter2)) and ("FILT_BBF_J"  in str(filter1)): lambda1, lambda2 = 1.190, 1.273
        elif ("FILT_DBF_K12" in str(filter2)) and ("FILT_BBF_Ks" in str(filter1)): lambda1, lambda2 = 2.110, 2.251
        else:
            # Single-BB filter fallback (rare). Keep both halves same λ.
            if "FILT_BBF_Y" in str(filter1):   lambda1 = lambda2 = 1.043
            elif "FILT_BBF_J" in str(filter1): lambda1 = lambda2 = 1.245
            elif "FILT_BBF_H" in str(filter1): lambda1 = lambda2 = 1.625
            elif "FILT_BBF_Ks" in str(filter1):lambda1 = lambda2 = 2.182
            else:                               lambda1 = lambda2 = 0.0

        # --- 5b) Process frames within this file ---
        for i in range(numc):
            print(f"\r file, frame = {j}, {i}", end="", flush=True)
            pa = pa1 + (pa2 - pa1) / max(1, numc) * i
            tstamp = itime  # keep IDL behavior

            # Centers per IDL:
            if getcen:
                # No center frames: start from image center; refine later via peakfind analog
                xc1 = yc1 = cc
                xc2 = yc2 = cc
            else:
                # Use MEDIAN centers for *every* science frame (IDL default)
                xc1, yc1 = xc1_med, yc1_med
                xc2, yc2 = xc2_med, yc2_med

            # Undither and center (IDL: shift_sub(..., -posx+dx, -posy+dy))
            dx1, dy1 = cc - xc1, cc - yc1
            dx2, dy2 = cc - xc2, cc - yc2

            # Slice halves and shift. scipy.ndimage.shift uses (shift_y, shift_x).
            left  = img[i, 0:sz,       0:sz]
            right = img[i, 0:sz,       sz:2*sz]
            img1  = shift(left,  (dy1 - posy, dx1 - posx), order=3)
            img2  = shift(right, (dy2 - posy, dx2 - posx), order=3)

            # Crop
            img1 = img1[pp0:pp0+szn, pp0:pp0+szn]
            img2 = img2[pp0:pp0+szn, pp0:pp0+szn]

            # If no center frames, refine on-the-fly (IDL: peakfind → re-center)
            if getcen:
                xc1p, yc1p = bscentrd(img1, ccn, ccn)
                xc2p, yc2p = bscentrd(img2, ccn, ccn)
                img1 = shift(img1, (ccn - yc1p, ccn - xc1p), order=3)
                img2 = shift(img2, (ccn - yc2p, ccn - xc2p), order=3)

            # Accumulate
            imc.append(img1); imc.append(img2)
            detside.extend([0, 1])
            pas.extend([pa, pa])
            dits.extend([dit, dit])
            stokes.extend([stoke12, stoke12])
            polfilts.extend([0, 0])  # IDL didn't store these; keep placeholders
            lambdas.extend([lambda1, lambda2])
            tstamps.extend([tstamp, tstamp])

    # --- 6) Finalize arrays and save ---
    imc      = np.asarray(imc)
    detside  = np.asarray(detside)
    pas      = -np.asarray(pas)  # IDL flips sign at end
    dits     = np.asarray(dits)
    stokes   = np.asarray(stokes)
    polfilts = np.asarray(polfilts)
    lambdas  = np.asarray(lambdas)
    tstamps  = np.asarray(tstamps)

    lambdas0 = np.unique(lambdas)
    nlambda  = len(lambdas0)

    # if cenbyfit: centerbyfit(imc)  # keep comment like IDL

    if outfile is None:
        outfile = "imcbasic.save"
    np.savez(
        outfile,
        imc=imc, detside=detside, pas=pas, lambdas=lambdas, dits=dits,
        normfac=normfac, stokes=stokes, polfilts=polfilts, tstamps=tstamps,
        lambdas0=lambdas0, nlambda=nlambda
    )


def sph_remove_bad_images(insavefile, inrad=None, outrad=None, fullres=False):
    """
    Remove bad images from a saved image cube based on RMS outliers.

    Parameters:
      insavefile : str - Filename of the saved .npz image cube.
      inrad : float, optional - Inner radius for region selection.
      outrad : float, optional - Outer radius for region selection.
      fullres : bool, optional - If True, return full outputs from 'get_good_images'.

    Returns:
      Updated .npz file with bad images removed.
      If fullres=True, returns a named tuple 'res' containing:
        - wg : Indices of good images.
        - wb : Indices of bad images.
        - rms : RMS values for all images.
        - med_rms : Median RMS value.
        - sig : Standard deviation of RMS values.
        - cut_factor : Final cut factor used.
    """
    import numpy as np

    res = None  # Default return value

    # Load data
    #data = np.load(insavefile)
    data = np.load(insavefile, allow_pickle=True)
    imc = data["imc"]


    # Extract necessary parameters
    pas = data.get("pas", None)
    dits = data.get("dits", None)
    lambdas = data.get("lambdas", None)
    lambdas0 = data.get("lambdas0", None)
    nlambda = data.get("nlambda", None)
    stokes = data.get("stokes", None)
    polfilts = data.get("polfilts", None)
    detside = data.get("detside", None)
    tstamps = data.get("tstamps", None)

    # Get unique wavelengths
    unique_lambdas = np.unique(lambdas) if lambdas is not None else np.array([0])

    wg_list, wb_list = [], []  # Lists to store indices of good and bad images


##modified_nav
    for lam in unique_lambdas:
        wl = np.where(lambdas == lam)[0]

        wg_sub = get_good_images(imc[wl], inrad=inrad, outrad=outrad, fullres=fullres)

        if fullres:
            res = wg_sub
            wg_sub = res.wg

        wg_sub = np.asarray(wg_sub, dtype=int)
        wg_sub = wg_sub[(wg_sub >= 0) & (wg_sub < len(wl))]

        good_idx = wl[wg_sub]
        bad_idx  = np.setdiff1d(wl, good_idx)

        wg_list.append(good_idx)
        wb_list.append(bad_idx)


    # Flatten lists of indices
    wg = np.concatenate(wg_list) if wg_list else np.array([])
    wb = np.concatenate(wb_list) if wb_list else np.array([])

    numg = len(wg)
    num = imc.shape[0]

    if numg < num:
        print(f"Removing {num - numg} bad images...")

        # Filter the arrays based on good image indices
        imc = imc[wg, :, :]
        dits = dits[wg] if dits is not None else None
        pas = pas[wg] if pas is not None else None
        lambdas = lambdas[wg] if lambdas is not None else None
        stokes = stokes[wg] if stokes is not None else None
        polfilts = polfilts[wg] if polfilts is not None else None
        detside = detside[wg] if detside is not None else None
        tstamps = tstamps[wg] if tstamps is not None else None

        # Save filtered data
        np.savez(
            insavefile,
            imc=imc, pas=pas, lambdas=lambdas, dits=dits, lambdas0=lambdas0, nlambda=nlambda,
            stokes=stokes, polfilts=polfilts, detside=detside, tstamps=tstamps
        )

        print(f"Updated {insavefile}: {numg} good images retained.")

    else:
        print(f"No bad images detected in {insavefile}.")

    return res if fullres else None
    


def get_good_images(imc, inrad=None, outrad=None, cut_factor=None, nodisp=False, fullres=False):
    """
    Flag bad images in a cube by checking the RMS near the center after subtracting the median.

    Parameters:
      imc : numpy.ndarray - 3D array (num_images, height, width).
      inrad : float, optional - Inner radius for region selection. Default: 0.
      outrad : float, optional - Outer radius for region selection. Default: min(10, sz/4).
      cut_factor : float, optional - Factor to determine rejection threshold.
      nodisp : bool, optional - If True, suppresses plotting.
      fullres : bool, optional - If True, return all results in a named tuple.

    Returns:
      wg : numpy.ndarray - Indices of good images.
      If fullres=True, returns a named tuple 'res' containing:
        - wg : Indices of good images.
        - wb : Indices of bad images.
        - rms : RMS values for all images.
        - med_rms : Median RMS value.
        - sig : Standard deviation of RMS values.
        - cut_factor : Final cut factor used.
    """
    import numpy as np
    from scipy.ndimage import median_filter
    from collections import namedtuple
    from .zfuncs import zplot2, getrgn

    res = namedtuple("res", ["wg", "wb", "rms", "med_rms", "sig", "cut_factor"])

    num, sz, _ = imc.shape  # Get number of images and image size

    # Default values for inrad and outrad
    if outrad is None:
        outrad = min(10, sz / 4.0)
    if inrad is None:
        inrad = 0

    # Compute median image across the cube
    med = np.median(imc, axis=0)

    # Compute RMS for each image
    rms = np.zeros(num, dtype=np.float32)

    for i in range(num):
        img = imc[i, :, :]
        temp = img - med
        mask = getrgn(med, inrad, outrad)  # Define the region for RMS calculation
        rms[i] = np.std(temp[mask])

    # Handle non-finite values
    rms[~np.isfinite(rms)] = -999

    # Compute median and standard deviation
    med_rms = np.median(rms)
    sig = np.std(rms)

    # Determine the cutoff factor if not provided
    if cut_factor is None:
        cut_factor = (med_rms + 2.0 * sig) / med_rms

    frms = np.abs(rms) / med_rms

    # Define good and bad images
    wg = np.where(frms <= cut_factor)[0]  # Good images
    wb = np.where(frms > cut_factor)[0]   # Bad images

    print(f"Number of good images: {len(wg)}")
    print(f"Median RMS: {med_rms}, Sigma: {sig}, Cut Factor: {cut_factor}")

    inx = np.arange(num)  # Equivalent to IDL's indgen(num)
    if not nodisp:
        zplot2(
            x=[inx, inx[wg]],
            y=[rms, rms[wg]],
            xtit="Image Index", ytit="RMS",
            legend=["All points", "Good points"],
            ylog=True,
            linestyle=[None, None],  # Pass as a valid list
            dotsize=[3, 5]
            )

    if fullres:
        return res(wg, wb, rms, med_rms, sig, cut_factor)

    return wg


def sph_rescale_ref_psf(science_file, reference_file, output_file):
    """
    Rescale the reference PSFs to match the science PSF profiles.

    Parameters:
      science_file : str - Filename of the science image cube (.npz).
      reference_file : str - Filename of the reference image cube (.npz).
      output_file : str - Filename to save the rescaled reference images.

    Returns:
      Saves a new .npz file with the rescaled reference images.
    """
    # Load science and reference image cubes
    import numpy as np
    
    s = np.load(science_file, allow_pickle=True)
    r = np.load(reference_file, allow_pickle=True)

    # Extract images
    imc_s = s["imc"]
    imc_r = r["imc"]
    #imc_s = s.imc
    #imc_r = r.imc

    nums = imc_s.shape[0]  # Number of science images
    numr = imc_r.shape[0]  # Number of reference images

    smed = np.median(imc_s, axis=0)  # Median science PSF
    rmed = np.median(imc_r, axis=0)  # Median reference PSF

    sz = smed.shape[0]
    cc = (sz - 1) / 2.0

    r0, r1 = 8, 66  # Coro edge and AO ring

    # Generate a circular distance map
    dd = zf.dist_circle(sz, cc)
    dd = np.maximum(dd, r0)

    # Define different regions in the PSF
    w1 = zf.getrgn(smed, r0 - 1, r0 + 1)  # Coro edge
    w2 = zf.getrgn(smed, 60, 90)          # AO-ring region
    w3 = zf.getrgn(smed, r1 - 1, r1 + 1)  # AO-ring
    wb = zf.getrgn(smed, 120, 140)        # Background region
    wm = np.logical_or(w1, w2)            # Matching region

    # Initialize output reference image cube
    rimc = np.copy(imc_r)

    # Iterate over reference images
    for j in range(numr):
        simg = smed.copy()
        rimg = rmed.copy()

        # Subtract background median
        simg -= np.median(simg[wb])
        rimg -= np.median(rimg[wb])

        # Compute power-law scaling factor
        f1, f2 = np.median(simg[w1]), np.median(simg[w3])
        p = -np.log(f1 / f2) / np.log(r0 / r1)

        # Compute scaling factors for coro-edge and AO ring
        c1, c2 = np.sum(simg[w1]), np.sum(rimg[w1])
        b1, b2 = np.std(simg[w2]), np.std(rimg[w2])

        # Rescale reference PSF to match science PSF
        pmap1 = (c1 - b1) * (dd / r0) ** (-p) + b1
        pmap2 = (c2 - b2) * (dd / r0) ** (-p) + b2
        pmap = pmap1 / pmap2
        rimg *= pmap

        # Compute the difference image
        diff = simg - rimg

        # Plot radial profiles
        rp1 = zf.radprof(simg, med=True, dia=2)
        rp2 = zf.radprof(rimg, med=True, dia=2)
        rpd = zf.radprof(diff, med=True, dia=2)

        rr = np.arange(len(rp1))  # Generate x-values as indices

        zf.zplot2(
            x=[rr,rr,rr],  
            y=[rp1, rp2, rpd],  
            legend=["Science", "Reference", "Difference"], 
            xlog=True, xtit="Radius", ytit="Profile"
            )

        rimc[j] = rimg  # Store rescaled reference image

    # Save the updated reference image cube
    #np.savez(output_file, imc=rimc, **r)
    r_dict = dict(r)  # Convert NpzFile to a regular dictionary
    r_dict.pop("imc", None)  # Now remove 'imc' safely
    np.savez(output_file, imc=rimc, **r_dict)  # Save filtered data

    print(f"Updated reference PSF saved to {output_file}")
