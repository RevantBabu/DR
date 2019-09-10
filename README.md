**Command to extract spike times from .mat files:**

$ python mat_to_py/position.py {day} {epoch} {tetrode} {cell}

The data in raw format is stored in data/raw/hc_13 folder. The above script uses this
data to produce the results at data/processed/hc_13/{day}/{epoch}/{tetrode}\_{cell}

**Command to get cell information summary:**

$ python mat_to_py/mat.py

**Command to generate VR distance:**

$ python core/VR.py {day} {epoch} {cell_id} {start_time} {end_time}

**Command to run diffusion map:**

$ python core/diffusion_lib.py {day} {epoch} {cell_id} {start_time} {end_time}