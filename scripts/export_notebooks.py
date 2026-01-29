#!/usr/bin/env python3
# this script exports the jupyter notebooks to html files

import subprocess
from pathlib import Path

# list of notebook files we need to export
notebook_files_list = [
    '01_phase1_data_build.ipynb',
    '02_phase2_retrieval_eval.ipynb', 
    '03_phase3_bias_audit.ipynb',
    '04_phase4_classifier_head.ipynb',
    '05_phase5_mitigations.ipynb'
]

# figure out where everything is
script_location = Path(__file__).parent.parent
notebooks_directory = script_location / 'notebooks'
reports_directory = script_location / 'reports'

# make reports folder if it doesnt exist
if not reports_directory.exists():
    reports_directory.mkdir(exist_ok=True)

print("Starting export of notebooks to HTML")
print(f"Saving reports to: {reports_directory}")
print()

# go through each notebook
for notebook_file in notebook_files_list:
    # get the full path to the notebook
    full_notebook_path = notebooks_directory / notebook_file
    
    # check if file exists first
    if not full_notebook_path.exists():
        print(f"Warning: Skipping {notebook_file} (file not found)")
        continue
    
    print(f"Converting {notebook_file}")
    
    # figure out output filename
    html_filename = notebook_file.replace('.ipynb', '.html')
    output_file_path = reports_directory / html_filename
    
    try:
        # run the jupyter nbconvert command
        result = subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'html',
            '--no-input',  # hide code cells, show only outputs and markdown
            '--output', str(output_file_path.resolve()),
            str(full_notebook_path.resolve())
        ], check=True, capture_output=True)
        
        print(f"Successfully completed and saved to {html_filename}")
    except subprocess.CalledProcessError as error:
        print(f"Error converting {notebook_file}: {error}")
    except FileNotFoundError:
        print(f"Error: jupyter nbconvert not found. Try running: pip install nbconvert")
        break

print()
print("Check the reports/ folder for HTML files")
print("You can open them in your web browser")
