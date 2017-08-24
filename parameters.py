import numpy as np
import tensorflow as tf
import os

print("--> Loading parameters...")

global par, analysis_par

"""
Independent parameters
"""
par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'debug_model'           : False,
    'load_previous_model'   : False,
    'analyze_model'         : False,

    # Network configuration
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'var_delay'             : False,
    'catch_trials'          : False,     # Note that turning on var_delay implies catch_trials

    # Network shape
    'num_motion_tuned'      : 36,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 40,
    'n_reflect'             : 39,
    'n_output'              : 3,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 5e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1,         # Usually 1

    # Variance values
    'clip_max_grad_val'     : 0.25,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.01,
    'noise_rnn_sd'          : 0.25,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 2,        # magnitutde scaling factor for von Mises
    'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-6,

    # Performance thresholds
    'stop_perf_th'          : 0.99,
    'stop_error_th'         : 0,

    # Training specs
    'batch_train_size'      : 128,
    'num_batches'           : 8,
    'num_iterations'        : 20,
    'iters_between_outputs' : 10,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 100,
    'fix_time'              : 100,
    'sample_time'           : 200,
    'delay_time'            : 400,
    'test_time'             : 200,
    'rule_onset_time'       : 1900,
    'rule_offset_time'      : 2100,
    'variable_delay_max'    : 500,
    'mask_duration'         : 80,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.15,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model.ckpt',
    'ckpt_load_fn'          : 'model.ckpt'
}

"""
Parameters to be used before running analysis
"""
analysis_par = {
    'analyze_model'         : True,
    'load_previous_model'   : True,
    'num_iterations'        : 1,
    'num_batches'           : 1,
    'batch_train_size'      : 1024*2,
    'var_delay'             : False,
    'dt'                    : 20,
    'learning_rate'         : 0,
    'catch_trial_pct'       : 0,
}

"""
Parameters to be used after running analysis
"""
revert_analysis_par = {
    'analyze_model'         : False,
    'load_previous_model'   : False,
    'num_iterations'        : 1500,
    'num_batches'           : 32,
    'batch_train_size'      : 32,
    'var_delay'             : True,
    'dt'                    : 20,
    'learning_rate'         : 5e-3,
    'catch_trial_pct'       : 0.15,
    'delay_time'            : 1000
}


"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for key, val in updates.items():
        par[key] = val
        print(key, val)

    update_trial_params()
    update_dependencies()

def update_trial_params():

    """
    Update all the trial parameters given trial_type
    """

    if par['trial_type'] == 'DMS':
        par['num_rules'] = 1
        par['num_rule_tuned'] = 0

    if par['trial_type'] == 'DMRS45':
        par['rotation_match'] = 45

    elif par['trial_type'] == 'DMRS90':
        par['rotation_match'] = 90

    elif  par['trial_type'] == 'DMRS180':
        par['rotation_match'] = 180

    elif par['trial_type'] == 'dualDMS':
        par['catch_trial_pct'] = 0
        par['num_receptive_fields'] = 2
        par['num_rules'] = 2
        par['probe_trial_pct'] = 0
        par['probe_time'] = 10
        par['num_rule_tuned'] = 12
        par['spike_cost'] = 0.005
        #par['num_iterations'] = 1500
        analysis_par['probe_trial_pct'] = 0.5

    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        par['catch_trial_pct'] = 0
        par['match_test_prob'] = 0.5
        par['max_num_tests'] = 3
        par['delay_time'] = 3000
        par['ABBA_delay'] = int(par['delay_time']/par['max_num_tests']/2)
        par['repeat_pct'] = 0
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif par['trial_type'] == 'DMS+DMRS' or par['trial_type'] == 'DMS+DMRS_early_cue':
        par['rotation_match'] = [0, 90]
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        if par['trial_type'] == 'DMS+DMRS':
            par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 500
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 700
        else:
            par['rule_onset_time'] = par['dead_time']
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time']

    elif par['trial_type'] == 'DMS' or par['trial_type'] == 'DMC':
        pass

    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])



    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = par['dt']/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']


    # General event profile info
    #par['name_of_stimulus'], par['date_stimulus_created'], par['author_of_stimulus_profile'] = get_profile(par['profile_path'])
    # List of events that occur for the network
    #par['events'] = get_events(par['profile_path'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    if par['trial_type'] == 'dualDMS':
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    sd = 0.1

    par['u0'] = np.tril(np.random.normal(0, sd, (par['n_hidden'], par['n_reflect'])))
    norms = np.linalg.norm(par['u0'], axis=0)
    par['u0'] = np.float32(1/norms*par['u0'])

    par['w_in0'] = np.float32(np.random.normal(0, sd, (par['n_hidden'], par['n_input'])))
    par['w_out0'] = np.float32(np.random.normal(0, sd, (par['n_output'], par['n_hidden'])))
    par['b_rnn0'] = np.zeros((par['n_hidden'], 1), dtype=np.float32)
    par['b_out0'] = np.zeros((par['n_output'], 1), dtype=np.float32)

    # used to calculate WY
    par['triu'] =  np.triu(np.ones((par['n_reflect'], par['n_reflect']), dtype=np.float32), 1)
    par['diag'] =  np.eye((par['n_reflect']), dtype=np.float32)

    par['u_mask'] = np.tril(np.ones((par['n_hidden'], par['n_reflect']), dtype=np.float32))


update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")
