import os
from pathlib import Path
import sys
import logging

def set_path():
    #root
    project_root=Path(__file__).parent
    #root parent
    root_parent=project_root.parent

    #others
    project_dir={
        "root_parent":root_parent,
        "root":project_root,
        "games":os.path.join(project_root,'games'),
        "methods":os.path.join(project_root,'methods'),
        "ma_methods":os.path.join(project_root,'ma_methods'),
        "model":os.path.join(project_root,'model'),
        "distributions":os.path.join(project_root,'distributions'),
    }
    
    #add to environment variable
    for path in project_dir.values():
        if path not in sys.path:
            sys.path.insert(0,str(path))


