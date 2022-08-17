#!/usr/bin/env python3

import numpy as np
import pdbfixer
import openmm

import logging

import sys
import os
import glob
import time
import csv
import stat
import traceback
from pathlib import Path
import pickle

import platform
import dask.config
from distributed import Client, Worker, as_completed, get_worker

#######################################
### PRE-PROCESSING FUNCTIONS
#######################################

def fix_protein(pdb_file, root_output_path = Path('./')):
    """ Check the structure model for any fixes before sending it to be parameterized and minimized.
    INPUT:
        pdb_file: path object associated with the pdb structure file that is to be "fixed" by adding missing atoms (i.e. hydrogens). 
        output_path: path object pointing to where the "fixed" pdb file will be written. 
        logger_file: file object within which logging information is being written to for this specific protein.
    RETURNS:
        output_dict: dictionary with all necessary output keys
    """
    # prepping the output dictionary object with initial values
    output_dict = {'task': 'preprocessing', 
                   'nodeID': platform.node(), 
                   'workerID': get_worker().id,
                   'start_time': time.time(),
                   'output_pdb_file': None,
                   'logger_file': None,
                   'stop_time' : None,
                   'return_code': None}

    # making directory for output
    output_path = root_output_path / pdb_file.parent.name     # NOTE: based on an assumed directory organization
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    
    file_root = str(output_path / pdb_file.stem)
    # stashing
    output_dict['output_pdb_file'] = Path(file_root + '_fixed.pdb')
    output_dict['logger_file'] = Path(file_root + '.log')
    
    # setting up the individual run's logging file
    prep_logger = setup_logger('minimization_logger', output_dict['logger_file'])
    prep_logger.info(f"Pre-processing being performed on {output_dict['nodeID']} with {output_dict['workerID']}.")
    prep_logger.info(f'     Checking {pdb_file!s} for any required fixes (missing hydrogens and other atoms, etc).')
    
    try:
        with open(pdb_file,'r') as pdb, open(output_dict['output_pdb_file'],'w') as save_file:
            fixer = pdbfixer.PDBFixer(pdbfile=pdb)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms(seed=0)
            fixer.addMissingHydrogens()
            prep_logger.info(f"        Saving {output_dict['output_pdb_file']!s}.")
            openmm.app.pdbfile.PDBFile.writeFile(fixer.topology,fixer.positions,file=save_file)
        return_code = 0
    
    except Exception as e:
        prep_logger.info(f'For {pdb_file!s}: pre-processing code, fix_protein, failed with the following exception:\n{e}\n')
        return_code = 1

    finally:
        # stashing
        output_dict['return_code'] = return_code
        output_dict['stop_time'] = time.time()
        if output_dict['return_code'] == 0:
            prep_logger.info(f"Finished pre-processing {pdb_file!s}. This took {output_dict['stop_time']-output_dict['start_time']} seconds.\n")
        
        clean_logger(prep_logger)
        return output_dict


#######################################
### PROCESSING FUNCTIONS
#######################################

def will_restrain(atom: openmm.app.topology.Atom, rset: str) -> bool:
  """Returns True if the atom will be restrained by the given restraint set."""

  if rset == "non_hydrogen":
    return atom.element.name != "hydrogen"
  elif rset == "c_alpha":
    return atom.name == "CA"


def _add_restraints(
    system,             #system: openmm.System,
    reference_pdb,      #reference_pdb: openmm.PDBFile,
    stiffness,          #stiffness: openmm.unit.Unit,
    rset,               #rset: str,
    exclude_residues):  #exclude_residues: Sequence[int]):
  """Adds a harmonic potential that restrains the end-to-end distance."""
  
  assert rset in ["non_hydrogen", "c_alpha"]

  force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
  force.addGlobalParameter("k", stiffness)
  for p in ["x0", "y0", "z0"]:
    force.addPerParticleParameter(p)

  for i, atom in enumerate(reference_pdb.topology.atoms()):
    if atom.residue.index in exclude_residues:
      continue
    if will_restrain(atom, rset):
      force.addParticle(i, reference_pdb.positions[i])
  system.addForce(force)


def run_minimization(pdb_file, logger_file, openmm_dictionary = {}):
    """run preparation of OpenMM simulation object and subsequent energy minimization
    INPUT:
        pdb_file: path object associated with the "fixed" pdb structure file that is ready for simulation. 
        logger_file: string pointing to a log file within which logging information is being written for this specific protein.
        openmm_dictionary: dictionary object that contains all relevant parameters used for preparation and simulations in OpenMM; keys: 
            "forcefield": string, denotes the force field xml file to be used to prep parameters for the structure. Acceptable values: "amber99sb.xml", "amber14/protein.ff14SB.xml", "amber14/protein.ff15ipq.xml", "charmm36.xml", or others. NOTE: currently only accepts a single string so won't be able to include an implicit solvent model as an additional force field file. 
            "exclude_residues": list of residue indices to be ignored when setting restraints.
            "restraint_set": string used to denote which atoms within the structure are restrained.
            "restraint_stiffness": float, sets the restraint spring constant (units: energy_units length_units **-2) to be applied to all atoms defined in the restraint_set variable. 
            "max_iterations": int, the maximum number of minimization iterations that will be performed; default = 0, no limit to number of minimization calculations
            "energy_tolerance": float, the energy tolerance cutoff, below which, a structure is considered acceptably energy-minimized
            "fail_attempts": int, number of minimization attempts to try before giving up
            "energy_units": openmm unit object for energy
            "length_units": openmm unit object for length
    RETURNS:
        output_dict: dictionary containing all relevant info and objects for the run. Includes:
            'task': string that describes this function's purpose
            'nodeID', 'workerID': information about hardware being used for this calculation
            'output_pdb_file': path object pointing to the final energy minimized structure 
            'start_time', 'stop_time': floats, time values
            'return_code': integer value denoting the success of this function. 
    """
    # prepping the output dictionary object with initial values
    output_dict = {'task': 'processing', 
                   'nodeID': platform.node(), 
                   'workerID': get_worker().id,
                   'start_time': time.time(),
                   'logger_file': logger_file,
                   'output_pdb_file': None,
                   'stop_time' : None,
                   'return_code': None}

    out_file_path = logger_file.parent / logger_file.stem
    # setting up the individual run's logging file
    proc_logger = setup_logger('minimization_logger', logger_file)
    proc_logger.info(f"Prepping OpenMM simulation object and running energy minimization on {output_dict['nodeID']} with {output_dict['workerID']}.")
   
    # gathering openmm parameters
    forcefield          = openmm_dictionary['forcefield']
    exclude_residues    = openmm_dictionary['exclude_residues']
    restraint_set       = openmm_dictionary['restraint_set']
    restraint_stiffness = openmm_dictionary['restraint_stiffness']
    openmm_platform     = openmm_dictionary['openmm_platform']
    max_iterations      = openmm_dictionary['max_iterations']
    energy_tolerance    = openmm_dictionary['energy_tolerance']
    fail_attempts       = openmm_dictionary['fail_attempts']
    energy_units        = openmm_dictionary['energy_units']
    length_units        = openmm_dictionary['length_units']
    
    try: 
        start_time = time.time()
        proc_logger.info(f'Preparing the OpenMM simulation components:')
        # load pdb file into an openmm Topology and coordinates object.
        pdb = openmm.app.pdbfile.PDBFile(str(pdb_file))

        # set the FF and constraints objects.
        proc_logger.info(f'        Using {forcefield}.')
        force_field = openmm.app.forcefield.ForceField(forcefield)
        
        # prepare the restraints/constraints for the system.
        proc_logger.info(f'        Building HBond constraints as well as restraints on "{restraint_set}".')
        proc_logger.info(f'        Restraints have a spring constant of {restraint_stiffness} {energy_units} {length_units}^-2.')
        proc_logger.info(f'        ResIDs {exclude_residues} are not included in the restraint set.')
        constraints = openmm.app.HBonds
        system = force_field.createSystem(pdb.topology, constraints=constraints)
        stiffness = restraint_stiffness * energy_units / (length_units**2)
        if stiffness > 0. * energy_units / (length_units**2):
            _add_restraints(system, pdb, stiffness, restraint_set, exclude_residues)
        
        # create the integrator object. 
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)    # required to set this for prepping the simulation object; hard set because we won't be using it; still necessary to define for the creation of the simulation object
        
        # determine what hardware will be used to perform the calculations
        openmm_platform = openmm.Platform.getPlatformByName(openmm_platform)  
        
        # prep the simulation object
        simulation = openmm.app.Simulation(pdb.topology, system, integrator, openmm_platform)
        # set the atom positions for the simulation's system's topology
        simulation.context.setPositions(pdb.positions)
        # set the minimization energy convergence tolerance value
        tolerance = energy_tolerance * energy_units

        proc_logger.info(f'Finished prepping simulation components. This took {time.time()-start_time} seconds.')
        
        proc_logger.info(f'Running energy minimization.')
        start_time = time.time()
        
        # grab initial energies
        state = simulation.context.getState(getEnergy=True)
        einit = state.getPotentialEnergy().value_in_unit(energy_units)
        ## grab initial energies and positions
        #state = simulation.context.getState(getEnergy=True, getPositions=True)
        #einit = state.getPotentialEnergy().value_in_unit(energy_units)
        #posinit = state.getPositions(asNumpy=True).value_in_unit(length_units)
        
        proc_logger.info(f'        Starting energy: {einit} {energy_units}')
        
        # attempt to minimize the structure
        attempts = 0
        minimized = False
        while not minimized and attempts < fail_attempts:
            attempts += 1
            try:
                # running minimization
                simulation.minimizeEnergy(maxIterations=max_iterations,tolerance=tolerance)
                
                # return energies and positions
                state = simulation.context.getState(getEnergy=True, getPositions=True)
                efinal = state.getPotentialEnergy().value_in_unit(energy_units)
                positions = state.getPositions(asNumpy=True).value_in_unit(length_units)
                proc_logger.info(f'        Final energy: {efinal} {energy_units}')
                 
                # saving the final structure to a pdb
                out_file = Path(str(out_file_path) + '_min_%02d.pdb'%(attempts-1))
                with open(out_file,'w') as out_pdb:
                    openmm.app.pdbfile.PDBFile.writeFile(simulation.topology,positions,file=out_pdb)
                minimized = True
                return_code = 0
            except Exception as e:
                proc_logger.info(f'        Attempt {attempts}: {e}')

        proc_logger.info(f'        dE = {efinal - einit} {energy_units}')
        output_dict['output_pdb_file'] = out_file
        
        if not minimized:
            proc_logger.info(f"Minimization failed after {fail_attempts} attempts.\n")
            return_code = 1
    
        proc_logger.info(f'Finished running minimization. This took {time.time()-start_time} seconds.')

    except Exception as e:
        proc_logger.info(f'Processing code, run_minimization, failed with the following exception:\n{e}\n')
        return_code = 1

    finally:
        output_dict['stop_time'] = time.time()
        output_dict['return_code'] = return_code
        
        if output_dict['return_code'] == 0:
            proc_logger.info(f"Finished the processing task. This took {output_dict['stop_time']-output_dict['start_time']} seconds.\n")
        
        return output_dict


#######################################
### POST-PROCESSING FUNCTIONS
#######################################

def center(pdb_file, logger_file):
    """function to take a pdb structure file and translate the structure's center of geometry to the origin. 
    INPUT:
        pdb_file: path object associated with the pdb structure file that is to be translated
        logger_file: string pointing to a log file within which logging information is being written for this specific protein.
    RETURNS:
        out_pdb : path object associated with the new, centered pdb structure file (written during the function)
        start_time, stop_time: floats, time values
        return_code: integer value denoting the success of this function. 
    """
    # prepping the output dictionary object with initial values
    output_dict = {'task': 'post-processing', 
                   'nodeID': platform.node(), 
                   'workerID': get_worker().id,
                   #'workerID': worker.id,
                   'start_time': time.time(),
                   'logger_file': logger_file,
                   'output_pdb_file': None,
                   'stop_time' : None,
                   'return_code': None}

    # setting up the individual run's logging file
    post_logger = setup_logger('minimization_logger', logger_file)
    post_logger.info(f"Post-processing the energy minimized structure on {output_dict['nodeID']} with {output_dict['workerID']}.")
    post_logger.info(f"Removing CoG translation from the {pdb_file!s}.")
    
    out_file_path = logger_file.parent / logger_file.stem
    output_dict['output_pdb_file'] = Path(str(out_file_path) + '_centered.pdb')
    try:
        with open(pdb_file,'r') as pdb, open(output_dict['output_pdb_file'],'w') as out_pdb:
            line_lst = pdb.readlines()
            xyz_array = np.array([[float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in line_lst if line.split()[0] == 'ATOM'])
            avg_xyz = np.mean(xyz_array,axis=0)
            xyz_array -= avg_xyz
            post_logger.info(f'         Translated the system by {-avg_xyz}.')
            post_logger.info(f'         CoG now at {np.mean(xyz_array,axis=0)}.')
            count = 0 
            for line in line_lst:
                if line.split()[0] == 'ATOM':
                    out_pdb.write(line[:30] + '%8.3f%8.3f%8.3f'%(xyz_array[count,0],xyz_array[count,1],xyz_array[count,2]) + line[54:])
                    count += 1
                else:
                    out_pdb.write(line)
        return_code = 0

    except Exception as e:
        post_logger.info(f'For {pdb_file!s}: post-processing code, center, failed with the following exception:\n{e}\n')
        return_code = 1

    finally:
        output_dict['stop_time'] = time.time()
        output_dict['return_code'] = return_code
        if output_dict['return_code'] == 0:
            post_logger.info(f"Finished processing files {pdb_file!s} after {output_dict['stop_time']-output_dict['start_time']} seconds.\n")
        clean_logger(post_logger)
        
        return output_dict


#######################################
### DASK RELATED FUNCTIONS
#######################################

def get_num_workers(client):
    """ Get the number of active workers
    :param client: active dask client
    :return: the number of workers registered to the scheduler
    """
    scheduler_info = client.scheduler_info()

    return len(scheduler_info['workers'].keys())


def disconnect(client, workers_list):
    """ Shutdown the active workers in workers_list
    :param client: active dask client
    :param workers_list: list of dask workers
    """
    client.retire_workers(workers_list, close_workers=True)
    client.shutdown()


#######################################
### LOGGING FUNCTIONS
#######################################

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s    %(levelname)s       %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def clean_logger(logger):
    """To cleanup the logger instances once we are done with them"""
    for handle in logger.handlers:
        handle.flush()
        handle.close()
        logger.removeHandler(handle)


def append_timings(csv_writer, file_object, nodeID, workerID, start_time, stop_time, task, return_code, file_path):
    """ append the task timings to the CSV timings file
    :param csv_writer: CSV to which to append timings
    :param hostname: on which the processing took place
    :param worker_id: of the dask worker that did the processing
    :param start_time: start time in *NIX epoch seconds
    :param stop_time: stop time in same units
    :param end_string: that was processed
    """
    csv_writer.writerow({'nodeID'     : nodeID,
                         'workerID'   : workerID,
                         'start_time' : start_time,
                         'stop_time'  : stop_time,
                         'task'       : task,
                         'return_code': return_code,
                         'file_path'  : file_path})
    file_object.flush()


#######################################
### MAIN
#######################################

if __name__ == '__main__':
   
    client = Client(scheduler_file=sys.argv[1],timeout=5000,name='all_tsks_client')
    with open(sys.argv[2],'r') as structures_file:
        structure_list = [Path(line.strip()) for line in structures_file.readlines() if line[0] != '#']
    
    root_output_path = Path(sys.argv[3])
    
    with open(sys.argv[4],'rb') as pickle_file:
        openmm_dictionary = pickle.load(pickle_file)
    # check if all parameters have been explicitly defined; if they haven't, fill in with default values. # NOTE apply this code.

    # set up timing log file
    timings_file = open(sys.argv[5],'w')
    timings_csv = csv.DictWriter(timings_file,['nodeID','workerID','start_time','stop_time','task','return_code','file_path'])
    timings_csv.writeheader()

    ### CLEANER IDEA 
    #task_futures = client.map(preprocessing_pipeline,structure_list, root_output_path = root_output_path, openmm_dictionary = openmm_dictionary, pure = False, resources={'CPU':1})
    #futures_bucket = as_completed(task_futures)
    #for i, finished_task in enumerate(futures_bucket):
    #    dictionary_of_objects = finished_task.results()
    #    #future = client.submit(dictionary_of_objects['next_task_function'], variables, resources = {dictionary_result} )
    #    futures_bucket.add(future)


    # submit pre-processing tasks
    task_futures = client.map(fix_protein,structure_list, root_output_path = root_output_path, pure = False, resources={'CPU':1})
    # checking preprocessing tasks
    preprocess_ac = as_completed(task_futures)
    processing_futures = []
    for i, finished_task in enumerate(preprocess_ac):
        prep_results_dictionary = finished_task.result()
        
        # collection of timing and metadata info
        append_timings(timings_csv,timings_file,prep_results_dictionary['nodeID'],prep_results_dictionary['workerID'],prep_results_dictionary['start_time'],prep_results_dictionary['stop_time'],prep_results_dictionary['task'],prep_results_dictionary['return_code'],str(prep_results_dictionary['output_pdb_file'].parent))
        
        # check to see if preprocessing ran successfully
        if prep_results_dictionary['return_code'] == 0:
            # submit processing task to GPU resources
            processing_future = client.submit(run_minimization, prep_results_dictionary['output_pdb_file'], prep_results_dictionary['logger_file'], openmm_dictionary = openmm_dictionary, pure = False, resources={'GPU':1})
            processing_futures.append(processing_future)
    
    # checking processing tasks
    processing_ac = as_completed(processing_futures)
    postprocessing_futures = []
    for i, finished_task in enumerate(processing_ac):
        proc_results_dictionary = finished_task.result()
        
        # collection of timing and metadata info
        append_timings(timings_csv,timings_file,proc_results_dictionary['nodeID'],proc_results_dictionary['workerID'],proc_results_dictionary['start_time'],proc_results_dictionary['stop_time'],proc_results_dictionary['task'],proc_results_dictionary['return_code'],str(proc_results_dictionary['output_pdb_file']))
        
        # check to see if processing ran successfully
        if proc_results_dictionary['return_code'] == 0:
            # submit postprocessing task to CPU resources
            postprocessing_future = client.submit(center, proc_results_dictionary['output_pdb_file'], proc_results_dictionary['logger_file'], pure=False, resources={'CPU':1})
            postprocessing_futures.append(postprocessing_future)

    # checking postprocessing tasks
    postprocessing_ac = as_completed(postprocessing_futures)
    for i, finished_task in enumerate(postprocessing_ac):
        #final_pdb_file, start_time, stop_time, return_code = finished_task.result()
        post_results_dictionary = finished_task.result()
        
        # collection of timing and metadata info
        append_timings(timings_csv,timings_file,post_results_dictionary['nodeID'],post_results_dictionary['workerID'],post_results_dictionary['start_time'],post_results_dictionary['stop_time'],post_results_dictionary['task'],post_results_dictionary['return_code'],str(post_results_dictionary['output_pdb_file']))

    sys.stdout.flush()  # Because Summit needs nudged
    sys.stderr.flush()
    
    client.shutdown()

