import os
import glob
import subprocess

def get_file_list(data_dir):
    pattern = os.path.join(data_dir, 'SPH*.[0-9][0-9][0-9].fits')
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
        
