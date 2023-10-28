# generic-python-project-template
generic-python-project-template

# generic-python-project-template
generic-python-project-template

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### to copy any file 

```bash
cp src/stage__00__template.py src/01_base_model_creation.py

```

### To rename any file 

```bash 
mv src/stage__00__template.py src/02_transfer_learning.py
```
