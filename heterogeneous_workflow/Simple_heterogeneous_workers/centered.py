#!/usr/bin/env python3

import numpy as np
import sys
from pathlib import Path

#import parmed
#import openmm
#import logging

#import time
#import os
#import stat
#import traceback
#import csv

#import logging_functions

pdb_file = Path(sys.argv[1])
path = pdb_file.parent

xyz_array = []
with open(pdb_file,'r') as pdb, open(path / str(pdb_file.stem+'_centered.pdb'),'w') as out_pdb:
    line_lst = pdb.readlines()
    xyz_array = np.array([[float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in line_lst if line.split()[0] == 'ATOM'])
    avg_xyz = np.mean(xyz_array,axis=0)
    print(avg_xyz)
    xyz_array -= avg_xyz
    count = 0 
    for line in line_lst:
        if line.split()[0] == 'ATOM':
            out_pdb.write(line[:30] + '%8.3f%8.3f%8.3f'%(xyz_array[count,0],xyz_array[count,1],xyz_array[count,2]) + line[54:])
            count += 1
        else:
            out_pdb.write(line)

#pdb = openmm.app.pdbfile.PDBFile(pdb_file)
#force_field = openmm.app.forcefield.ForceField('amber14/protein.ff14SB.xml')
#system = force_field.createSystem(pdb.topology)
#structure = parmed.openmm.load_topology(pdb.topology,system)
#parmed.tools.energy(structure,'omm platform CPU decompose').execute()

