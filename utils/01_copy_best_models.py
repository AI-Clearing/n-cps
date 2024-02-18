import os
from pathlib import Path
import pandas as pd
from glob import glob
import numpy as np
import shutil

# generated by notebooks/04_scaling.ipynb in the paper repository based on downloaded tb logs
CSV_WITH_BEST_AND_LAST_MODELS = "/home/dfilipiak/projects/ncps/utils/entropy.csv"
LOCAL_PATH = Path("/local_storage_1/dfilipiak/ncps/output/")
EXPORT_PATH = Path("/home/dfilipiak/projects/ncps/output/")
DONE_FLAG = ".best_and_last_snapshots_copied_to_home"           # TODO: warning - change the flag every time!

experiments = [Path(x) for x in glob("/local_storage_1/dfilipiak/ncps/output/*/*")]



df = pd.read_csv(CSV_WITH_BEST_AND_LAST_MODELS)

for p in experiments:

    experiment_name = f"{p.parent.name}/{p.name}"
    df_experiment = df[df["dir"] == experiment_name]
    done_flag_path = Path(LOCAL_PATH / experiment_name / "snapshot" / ".best_and_last_snapshots_copied_to_home" )    
    target_dir = EXPORT_PATH / experiment_name / "snapshot" / "snapshot"

    if len(df_experiment) < 1:
        print(f"[{experiment_name}]: Skipping (not found in reference DF, no files copied or deleted)\n")
    elif done_flag_path.exists():  
        print(f"[{experiment_name}]: Skipping (found done flag marked - files already copied and deleted)\n")
    elif not (LOCAL_PATH / experiment_name / "snapshot").exists():
        print(f"[{experiment_name}]: Skipping (dir {(LOCAL_PATH / experiment_name / 'snapshot')} not found)\n")
    else:
        files_to_copy = set(df_experiment["path_best"].unique().tolist() + df_experiment["path_last"].unique().tolist())
        
        for f in files_to_copy:
            # print(f"os.makedirs({target_dir}, exist_ok=True)")
            os.makedirs(target_dir, exist_ok=True)

            assert target_dir.is_dir()
            print(f"shutil.copy2({f}, {target_dir})") 
            shutil.copy2(f, target_dir)
            
        print(f'shutil.rmtree({LOCAL_PATH / experiment_name / "snapshot" / "snapshot"})')
        shutil.rmtree(LOCAL_PATH / experiment_name / "snapshot" / "snapshot")
        
        print(f'Path({done_flag_path}).touch()')
        done_flag_path.touch()
        print("\n")


# srun --partition=common --qos=16gpu4d -w asusgpu1 --pty bash
# conda activate semiseg
# python /home/dfilipiak/projects/ncps/utils/01_copy_best_models.py && du -h /local_storage_1/dfilipiak 
# 
        # LEFT
        # asusgpu2
        # /local_storage_1/dfilipiak/ncps/output/voc2.res18v3+.nCPS/n6-cpsw1.5-t0.0-nc0/snapshot/snapshot <- 0-79, brak eval?