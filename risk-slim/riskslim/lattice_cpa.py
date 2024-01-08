import time
import numpy as np
from cplex.callbacks import HeuristicCallback, LazyConstraintCallback
from cplex.exceptions import CplexError
from .bound_tightening import chained_updates, mc_chained_updates
from .defaults import DEFAULT_LCPA_SETTINGS
from .utils import print_log, validate_settings
from .heuristics import discrete_descent, mc_discrete_descent, sequential_rounding, mc_sequential_rounding
from .initialization import initialize_lattice_cpa
from .mip import add_mip_starts, convert_to_risk_slim_cplex_solution, mc_convert_to_risk_slim_cplex_solution, \
    create_risk_slim, set_cplex_mip_parameters
from .setup_functions import get_loss_bounds, mc_get_loss_bounds, setup_loss_functions, \
    setup_objective_functions, setup_penalty_parameters, mc_setup_penalty_parameters, mc_setup_objective_functions
from .solution_pool import SolutionPool, FastSolutionPool

DEFAULT_BOUNDS = {
    'objval_min': 0.0,
    'objval_max': float('inf'),
    'loss_min': 0.0,
    'loss_max': float('inf'),
    'L0_min': 0,
    'L0_max': float('inf'),
}


# LATTICE CPA FUNCTIONS
def run_lattice_cpa(data, constraints, settings=DEFAULT_LCPA_SETTINGS):
    """

    Parameters
    ----------
    data
    constraints
    settings

    Returns
    -------

    """

    mip_objects = setup_lattice_cpa(data, constraints, settings)

    model_info, mip_info, lcpa_info = finish_lattice_cpa(data, constraints, mip_objects, settings)

    return model_info, mip_info, lcpa_info


def setup_lattice_cpa(data, constraints, settings=DEFAULT_LCPA_SETTINGS):
    """

    Parameters
    ----------
    data, dict containing training data should pass check_data
    constraints, dict containing 'L0_min, L0_max, CoefficientSet'
    settings

    Returns
    -------
    mip_objects 
    
    """
    # process settings then split into manageable parts
    settings = validate_settings(settings, default_settings=DEFAULT_LCPA_SETTINGS)
    is_multiclass = settings["is_multiclass"]

    init_settings = {k.lstrip('init_'): settings[k] for k in settings if k.startswith('init_')}
    cplex_settings = {k.lstrip('cplex_'): settings[k] for k in settings if k.startswith('cplex_')}
    mc_settings = {k: settings[k] for k in settings if settings if k.startswith('mc_')}
    lcpa_settings = {k: settings[k] for k in settings if settings if not k.startswith(('init_', 'cplex_', 'mc_'))}

    # get handles for loss functions
    (Z,
     compute_loss,
     compute_loss_cut,
     compute_loss_from_scores,
     compute_loss_real,
     compute_loss_cut_real,
     compute_loss_from_scores_real) = setup_loss_functions(data=data,
                                                           coef_set=constraints['coef_set'],
                                                           L0_max=constraints['L0_max'],
                                                           loss_computation=settings['loss_computation'],
                                                           w_pos=settings['w_pos'],
                                                           is_multiclass=is_multiclass)

    # data
    if is_multiclass:
        N = Z[0].shape[0]
        F = Z[0].shape[-1]  # number of features
        K = Z[1].shape[-1]  # number of classes
        P = F * K
    else:
        N, P = Z.shape

    # trade-off parameters
    c0_value, C_0, L0_reg_ind, C_0_nnz = setup_penalty_parameters(c0_value=lcpa_settings['c0_value'],
                                                                  coef_set=constraints['coef_set'])
    if is_multiclass:
        mc_c0_value, mc_C_0, mc_L0_reg_ind, mc_C_0_nnz = mc_setup_penalty_parameters(
            mc_c0_value=mc_settings['mc_c0_value'],
            F=F)

    # major components
    if is_multiclass:
        (get_objval, get_L0_norm, get_L0_penalty, get_alpha, get_L0_penalty_from_alpha,
         mc_get_L0_norm, mc_get_L0_penalty, mc_get_beta, mc_get_L0_penalty_from_beta) = \
            mc_setup_objective_functions(compute_loss, L0_reg_ind, C_0_nnz, mc_L0_reg_ind, mc_C_0_nnz)
    else:
        (get_objval,
         get_L0_norm,
         get_L0_penalty,
         get_alpha,
         get_L0_penalty_from_alpha) = setup_objective_functions(compute_loss, L0_reg_ind, C_0_nnz)

    rho_lb = np.array(constraints['coef_set'].lb)
    rho_ub = np.array(constraints['coef_set'].ub)
    L0_min = constraints['L0_min']
    L0_max = constraints['L0_max']

    if is_multiclass:
        mc_L0_min = constraints['mc_L0_min']
        mc_L0_max = constraints['mc_L0_max']

        def is_feasible(rho, L0_min=L0_min, L0_max=L0_max, rho_lb=rho_lb, rho_ub=rho_ub, mc_L0_min=mc_L0_min,
                        mc_L0_max=mc_L0_max):
            nonzero_features = (np.count_nonzero(rho.reshape(mc_L0_reg_ind.shape[0], -1), axis=1) > 0)[mc_L0_reg_ind]
            number_of_features = np.sum(nonzero_features)
            return np.all(rho_ub >= rho) and np.all(rho_lb <= rho) and (
                    L0_min <= np.count_nonzero(rho[L0_reg_ind]) <= L0_max) and (
                           mc_L0_min <= number_of_features <= mc_L0_max)

    else:
        def is_feasible(rho, L0_min=L0_min, L0_max=L0_max, rho_lb=rho_lb, rho_ub=rho_ub):
            return np.all(rho_ub >= rho) and np.all(rho_lb <= rho) and (
                    L0_min <= np.count_nonzero(rho[L0_reg_ind]) <= L0_max)

    # compute bounds on objective value
    bounds = dict(DEFAULT_BOUNDS)
    bounds['L0_min'] = constraints['L0_min']
    bounds['L0_max'] = constraints['L0_max']
    if is_multiclass:
        bounds['loss_min'], bounds['loss_max'] = mc_get_loss_bounds(Z, rho_ub, rho_lb, L0_reg_ind, L0_max)
        bounds['mc_L0_min'] = constraints['mc_L0_min']
        bounds['mc_L0_max'] = constraints['mc_L0_max']
    else:
        bounds['loss_min'], bounds['loss_max'] = get_loss_bounds(Z, rho_ub, rho_lb, L0_reg_ind, L0_max)

    # initialize
    initial_pool = SolutionPool(P)
    initial_cuts = None

    # check if trivial solution is feasible, if so add it to the pool and update bounds
    trivial_solution = np.zeros(P)
    if is_feasible(trivial_solution):
        trivial_objval = compute_loss(trivial_solution)
        if lcpa_settings['initial_bound_updates']:
            bounds['objval_max'] = min(bounds['objval_max'], trivial_objval)
            bounds['loss_max'] = min(bounds['loss_max'], trivial_objval)
            if is_multiclass:
                bounds = mc_chained_updates(bounds, mc_C_0_nnz)
            else:
                bounds = chained_updates(bounds, C_0_nnz)

        initial_pool = initial_pool.add(objvals=trivial_objval, solutions=trivial_solution)

    # setup risk_slim_lp and risk_slim_mip parameters
    risk_slim_settings = {
        'C_0': c0_value,
        'coef_set': constraints['coef_set'],
        'tight_formulation': lcpa_settings['tight_formulation'],
        'drop_variables': lcpa_settings['drop_variables'],
        'include_auxillary_variable_for_L0_norm': lcpa_settings['include_auxillary_variable_for_L0_norm'],
        'include_auxillary_variable_for_objval': lcpa_settings['include_auxillary_variable_for_objval'],
    }
    risk_slim_settings.update(bounds)

    if is_multiclass:
        risk_slim_settings['mc_C_0'] = mc_c0_value
        risk_slim_settings['F'] = F
        risk_slim_settings['mc_C_0_nnz'] = mc_C_0_nnz
        risk_slim_settings['mc_L0_reg_ind'] = mc_L0_reg_ind

    # run initialization procedure
    if lcpa_settings['initialization_flag']:
        initial_pool, initial_cuts, initial_bounds = initialize_lattice_cpa(Z=Z,
                                                                            c0_value=lcpa_settings['c0_value'],
                                                                            constraints=constraints,
                                                                            bounds=bounds,
                                                                            settings=init_settings,
                                                                            risk_slim_settings=risk_slim_settings,
                                                                            cplex_settings=cplex_settings,
                                                                            compute_loss_from_scores=compute_loss_from_scores,
                                                                            compute_loss_real=compute_loss_real,
                                                                            compute_loss_cut_real=compute_loss_cut_real,
                                                                            compute_loss_from_scores_real=compute_loss_from_scores_real,
                                                                            get_objval=get_objval,
                                                                            get_L0_penalty=get_L0_penalty,
                                                                            is_feasible=is_feasible)

        if lcpa_settings['initial_bound_updates']:
            bounds.update(initial_bounds)
            risk_slim_settings.update(initial_bounds)

    # create risk_slim mip
    risk_slim_mip, risk_slim_indices = create_risk_slim(coef_set=constraints['coef_set'], input=risk_slim_settings,
                                                        is_multiclass=is_multiclass)
    risk_slim_indices['C_0_nnz'] = C_0_nnz
    risk_slim_indices['L0_reg_ind'] = L0_reg_ind

    if is_multiclass:
        risk_slim_indices['mc_C_0_nnz'] = mc_C_0_nnz
        risk_slim_indices['mc_L0_reg_ind'] = mc_L0_reg_ind

    # mip
    mip_objects = {
        'mip': risk_slim_mip,
        'indices': risk_slim_indices,
        'bounds': bounds,
        'initial_pool': initial_pool,
        'initial_cuts': initial_cuts,
    }

    return mip_objects


def finish_lattice_cpa(data, constraints, mip_objects, settings=DEFAULT_LCPA_SETTINGS):
    """

    Parameters
    ----------
    data, dict containing training data should pass check_data
    constraints, dict containing 'L0_min, L0_max, CoefficientSet'
    settings
    mip_objects output of setup_risk_slim
    
    Returns
    ------- 

    """

    # process settings then split into manageable parts
    settings = validate_settings(settings, default_settings=DEFAULT_LCPA_SETTINGS)
    is_multiclass = settings["is_multiclass"]

    cplex_settings = {k.lstrip('cplex_'): settings[k] for k in settings if k.startswith('cplex_')}
    mc_settings = {k: settings[k] for k in settings if settings if k.startswith('mc_')}
    lcpa_settings = {k: settings[k] for k in settings if settings if not k.startswith(('init_', 'cplex_', 'mc_'))}

    # unpack mip_objects from setup_risk_slim
    risk_slim_mip = mip_objects['mip']
    indices = mip_objects['indices']
    bounds = mip_objects['bounds']
    initial_pool = mip_objects['initial_pool']
    initial_cuts = mip_objects['initial_cuts']

    # get handles for loss functions
    # loss functions
    (Z,
     compute_loss,
     compute_loss_cut,
     compute_loss_from_scores,
     compute_loss_real,
     compute_loss_cut_real,
     compute_loss_from_scores_real) = setup_loss_functions(data=data,
                                                           coef_set=constraints['coef_set'],
                                                           L0_max=constraints['L0_max'],
                                                           loss_computation=settings['loss_computation'],
                                                           w_pos=settings['w_pos'],
                                                           is_multiclass=is_multiclass)

    # data
    if is_multiclass:
        N = Z[0].shape[0]
        F = Z[0].shape[-1]
        P = F * Z[1].shape[-1]
    else:
        N, P = Z.shape

    # trade-off parameter
    c0_value, C_0, L0_reg_ind, C_0_nnz = setup_penalty_parameters(c0_value=lcpa_settings['c0_value'],
                                                                  coef_set=constraints['coef_set'])

    if is_multiclass:
        mc_c0_value, mc_C_0, mc_L0_reg_ind, mc_C_0_nnz = mc_setup_penalty_parameters(
            mc_c0_value=mc_settings['mc_c0_value'],
            F=F)

    # major components
    if is_multiclass:
        (get_objval, get_L0_norm, get_L0_penalty, get_alpha, get_L0_penalty_from_alpha,
         mc_get_L0_norm, mc_get_L0_penalty, mc_get_beta, mc_get_L0_penalty_from_beta) = \
            mc_setup_objective_functions(compute_loss, L0_reg_ind, C_0_nnz, mc_L0_reg_ind, mc_C_0_nnz)
    else:
        (get_objval,
         get_L0_norm,
         get_L0_penalty,
         get_alpha,
         get_L0_penalty_from_alpha) = setup_objective_functions(compute_loss, L0_reg_ind, C_0_nnz)

    # constraints
    rho_lb = np.array(constraints['coef_set'].lb)
    rho_ub = np.array(constraints['coef_set'].ub)
    L0_min = constraints['L0_min']
    L0_max = constraints['L0_max']
    trivial_L0_max = np.sum(constraints['coef_set'].penalized_indices())

    if is_multiclass:
        mc_L0_min = constraints['mc_L0_min']
        mc_L0_max = constraints['mc_L0_max']

        def is_feasible(rho, L0_min=L0_min, L0_max=L0_max, rho_lb=rho_lb, rho_ub=rho_ub, mc_L0_min=mc_L0_min,
                        mc_L0_max=mc_L0_max):
            nonzero_features = (np.count_nonzero(rho.reshape(mc_L0_reg_ind.shape[0], -1), axis=1) > 0)[mc_L0_reg_ind]
            number_of_features = np.sum(nonzero_features)
            return np.all(rho_ub >= rho) and np.all(rho_lb <= rho) and (
                    L0_min <= np.count_nonzero(rho[L0_reg_ind]) <= L0_max) and (
                           mc_L0_min <= number_of_features <= mc_L0_max)

    else:
        def is_feasible(rho, L0_min=L0_min, L0_max=L0_max, rho_lb=rho_lb, rho_ub=rho_ub):
            return np.all(rho_ub >= rho) and np.all(rho_lb <= rho) and (
                    L0_min <= np.count_nonzero(rho[L0_reg_ind]) <= L0_max)

    risk_slim_mip = set_cplex_mip_parameters(risk_slim_mip, cplex_settings,
                                             display_cplex_progress=lcpa_settings['display_cplex_progress'])
    risk_slim_mip.parameters.timelimit.set(lcpa_settings['max_runtime'])

    # setup callback functions
    control = {
        'incumbent': np.repeat(np.nan, P),
        'upperbound': float('inf'),
        'bounds': dict(bounds),
        'lowerbound': 0.0,
        'relative_gap': float('inf'),
        'nodes_processed': 0,
        'nodes_remaining': 0,
        #
        'start_time': float('nan'),
        'total_run_time': 0.0,
        'total_cut_time': 0.0,
        'total_polish_time': 0.0,
        'total_round_time': 0.0,
        'total_round_then_polish_time': 0.0,
        #
        'cut_callback_times_called': 0,
        'heuristic_callback_times_called': 0,
        'total_cut_callback_time': 0.00,
        'total_heuristic_callback_time': 0.00,
        #
        # number of times solutions were updates
        'n_incumbent_updates': 0,
        'n_heuristic_updates': 0,
        'n_cuts': 0,
        'n_polished': 0,
        'n_rounded': 0,
        'n_rounded_then_polished': 0,
        #
        # total # of bound updates
        'n_update_bounds_calls': 0,
        'n_bound_updates': 0,
        'n_bound_updates_loss_min': 0,
        'n_bound_updates_loss_max': 0,
        'n_bound_updates_L0_min': 0,
        'n_bound_updates_L0_max': 0,
        'n_bound_updates_objval_min': 0,
        'n_bound_updates_objval_max': 0,
    }

    lcpa_cut_queue = FastSolutionPool(P)
    lcpa_polish_queue = FastSolutionPool(P)

    heuristic_flag = lcpa_settings['round_flag'] or lcpa_settings['polish_flag']

    if heuristic_flag:

        loss_cb = risk_slim_mip.register_callback(LossCallback)
        if is_multiclass:
            loss_cb.initialize(indices=indices,
                               control=control,
                               settings=lcpa_settings,
                               compute_loss_cut=compute_loss_cut,
                               get_alpha=get_alpha,
                               get_beta=mc_get_beta,
                               get_L0_penalty_from_alpha=get_L0_penalty_from_alpha,
                               get_L0_penalty_from_beta=mc_get_L0_penalty_from_beta,
                               initial_cuts=initial_cuts,
                               is_multiclass=is_multiclass,
                               polish_queue=lcpa_polish_queue
                               )
        else:
            loss_cb.initialize(indices=indices,
                               control=control,
                               settings=lcpa_settings,
                               compute_loss_cut=compute_loss_cut,
                               get_alpha=get_alpha,
                               get_L0_penalty_from_alpha=get_L0_penalty_from_alpha,
                               initial_cuts=initial_cuts,
                               is_multiclass=is_multiclass,
                               polish_queue=lcpa_polish_queue)

        heuristic_cb = risk_slim_mip.register_callback(PolishAndRoundCallback)
        if is_multiclass:
            mc_L0_min = constraints['mc_L0_min']
            mc_L0_max = constraints['mc_L0_max']
            trivial_mc_L0_max = F
            active_set_flag = mc_L0_max <= trivial_mc_L0_max
            polishing_handle = lambda rho: mc_discrete_descent(rho, Z, C_0, mc_C_0, rho_ub, rho_lb, get_L0_penalty,
                                                               mc_get_L0_penalty,
                                                               compute_loss=compute_loss,
                                                               compute_loss_from_scores=compute_loss_from_scores,
                                                               active_set_flag=active_set_flag)

            rounding_handle = lambda rho, cutoff: mc_sequential_rounding(rho, Z, C_0, mc_C_0,
                                                                         compute_loss_from_scores_real,
                                                                         get_L0_penalty, mc_get_L0_penalty, cutoff)
            heuristic_cb.initialize(indices=indices,
                                    control=control,
                                    settings=lcpa_settings,
                                    cut_queue=lcpa_cut_queue,
                                    polish_queue=lcpa_polish_queue,
                                    get_objval=get_objval,
                                    get_L0_norm=get_L0_norm,
                                    mc_get_L0_norm=mc_get_L0_norm,
                                    is_feasible=is_feasible,
                                    polishing_handle=polishing_handle,
                                    rounding_handle=rounding_handle,
                                    is_multiclass=is_multiclass)
        else:
            active_set_flag = L0_max <= trivial_L0_max
            polishing_handle = lambda rho: discrete_descent(rho, Z, C_0, rho_ub, rho_lb, get_L0_penalty,
                                                            compute_loss_from_scores, active_set_flag=active_set_flag)
            rounding_handle = lambda rho, cutoff: sequential_rounding(rho, Z, C_0, compute_loss_from_scores_real,
                                                                      get_L0_penalty, cutoff)
            heuristic_cb.initialize(indices=indices,
                                    control=control,
                                    settings=lcpa_settings,
                                    cut_queue=lcpa_cut_queue,
                                    polish_queue=lcpa_polish_queue,
                                    get_objval=get_objval,
                                    get_L0_norm=get_L0_norm,
                                    is_feasible=is_feasible,
                                    polishing_handle=polishing_handle,
                                    rounding_handle=rounding_handle)

    else:
        loss_cb = risk_slim_mip.register_callback(LossCallback)
        if is_multiclass:
            loss_cb.initialize(indices=indices,
                               control=control,
                               settings=lcpa_settings,
                               compute_loss_cut=compute_loss_cut,
                               get_alpha=get_alpha,
                               get_beta=mc_get_beta,
                               get_L0_penalty_from_alpha=get_L0_penalty_from_alpha,
                               get_L0_penalty_from_beta=mc_get_L0_penalty_from_beta,
                               initial_cuts=initial_cuts,
                               is_multiclass=is_multiclass)
        else:
            loss_cb.initialize(indices=indices,
                               control=control,
                               settings=lcpa_settings,
                               compute_loss_cut=compute_loss_cut,
                               get_alpha=get_alpha,
                               get_L0_penalty_from_alpha=get_L0_penalty_from_alpha,
                               initial_cuts=initial_cuts,
                               is_multiclass=is_multiclass)
    # attach solution pool
    if len(initial_pool) > 0:
        if lcpa_settings['polish_flag']:
            lcpa_polish_queue.add(initial_pool.objvals[0], initial_pool.solutions[0])
            # initialize using the polish_queue when possible since the CPLEX MIPStart interface is tricky
        else:
            risk_slim_mip = add_mip_starts(risk_slim_mip, indices, initial_pool,
                                           mip_start_effort_level=risk_slim_mip.MIP_starts.effort_level.repair)

        if lcpa_settings['add_cuts_at_heuristic_solutions'] and len(initial_pool) > 1:
            lcpa_cut_queue.add(initial_pool.objvals[1:], initial_pool.solutions[1:])

    # solve using lcpa
    control['start_time'] = time.time()
    risk_slim_mip.solve()
    control['total_run_time'] = time.time() - control['start_time']
    control.pop('start_time')

    # record mip solution statistics
    try:
        control['incumbent'] = np.array(risk_slim_mip.solution.get_values(indices['rho']))
        control['upperbound'] = risk_slim_mip.solution.get_objective_value()
        control['lowerbound'] = risk_slim_mip.solution.MIP.get_best_objective()
        control['relative_gap'] = risk_slim_mip.solution.MIP.get_mip_relative_gap()
        control['found_solution'] = True
    except CplexError:
        control['found_solution'] = False

    control['cplex_status'] = risk_slim_mip.solution.get_status_string()
    control['total_callback_time'] = control['total_cut_callback_time'] + control['total_heuristic_callback_time']
    control['total_solver_time'] = control['total_run_time'] - control['total_callback_time']
    control['total_data_time'] = control['total_cut_time'] + control['total_polish_time'] + control[
        'total_round_time'] + control['total_round_then_polish_time']

    # Output for Model
    model_info = {
        'c0_value': c0_value,
        'w_pos': settings['w_pos'],
        #
        'solution': control['incumbent'],
        'objective_value': get_objval(control['incumbent']) if control['found_solution'] else float('inf'),
        'loss_value': compute_loss(control['incumbent']) if control['found_solution'] else float('inf'),
        'optimality_gap': control['relative_gap'] if control['found_solution'] else float('inf'),
        #
        'run_time': control['total_run_time'],
        'solver_time': control['total_solver_time'],
        'callback_time': control['total_callback_time'],
        'data_time': control['total_data_time'],
        'nodes_processed': control['nodes_processed'],
        'cut_callback_times_called': control['cut_callback_times_called']
    }
    model_info.update(constraints)

    # Output for MIP
    mip_info = {
        'risk_slim_mip': risk_slim_mip,
        'risk_slim_idx': indices
    }

    # Output for LCPA
    lcpa_info = dict(control)
    lcpa_info['bounds'] = dict(bounds)
    lcpa_info['settings'] = dict(settings)

    print(
        f"\nOptimality Gap: {model_info['optimality_gap']}; Callback called: {model_info['cut_callback_times_called']}; Loss value: {model_info['loss_value']}; Nodes processed: {control['nodes_processed']}")
    return model_info, mip_info, lcpa_info


# CALLBACK FUNCTIONS
class LossCallback(LazyConstraintCallback):
    """
    This callback has to be initialized after construnction with initialize().

    LossCallback is called when CPLEX finds an integer feasible solution. By default, it will add a cut at this
    solution to improve the cutting-plane approximation of the loss function. The cut is added as a 'lazy' constraint
    into the surrogate LP so that it is evaluated only when necessary.

    Optional functionality:

    - add an initial set of cutting planes found by warm starting
      requires initial_cuts

    - pass integer feasible solutions to 'polish' queue so that they can be polished with DCD in the PolishAndRoundCallback
      requires settings['polish_flag'] = True

    - adds cuts at integer feasible solutions found by the PolishAndRoundCallback
      requires settings['add_cuts_at_heuristic_solutions'] = True

    - reduces overall search region by adding constraints on objval_max, l0_max, loss_min, loss_max
      requires settings['chained_updates_flag'] = True
    """

    def initialize(self, indices, control, settings, compute_loss_cut, get_alpha, get_L0_penalty_from_alpha,
                   get_beta=None, get_L0_penalty_from_beta=None, initial_cuts=None, cut_queue=None, polish_queue=None,
                   is_multiclass=False):

        assert isinstance(indices, dict)
        assert isinstance(control, dict)
        assert isinstance(settings, dict)
        assert callable(compute_loss_cut)
        assert callable(get_alpha)
        assert callable(get_L0_penalty_from_alpha)

        self.settings = settings  # store pointer to shared settings so that settings can be turned on/off during B&B
        self.control = control  # dict containing information for flow
        self.is_multiclass = is_multiclass

        # todo (validate initial cutting planes)
        self.initial_cuts = initial_cuts

        # indices
        self.rho_idx = indices['rho']
        self.cut_idx = indices['loss'] + indices['rho']
        self.alpha_idx = indices['alpha']
        self.beta_idx = indices['beta'] if 'beta' in indices else []
        self.L0_reg_ind = indices['L0_reg_ind']
        self.C_0_nnz = indices['C_0_nnz']
        self.mc_C_0_nnz = indices['mc_C_0_nnz'] if 'mc_C_0_nnz' in indices else []
        self.compute_loss_cut = compute_loss_cut
        self.get_alpha = get_alpha
        self.get_beta = get_beta if get_beta is not None else lambda rho: 0
        self.get_L0_penalty_from_alpha = get_L0_penalty_from_alpha
        self.get_L0_penalty_from_beta = get_L0_penalty_from_beta \
            if get_L0_penalty_from_beta is not None else lambda beta: 0

        # cplex has the ability to drop cutting planes that are not used. by default, we force CPLEX to use all cutting planes.
        self.loss_cut_purge_flag = self.use_constraint.purge if self.settings[
            'purge_loss_cuts'] else self.use_constraint.force

        # setup pointer to cut_queue to receive cuts from PolishAndRoundCallback
        if self.settings['add_cuts_at_heuristic_solutions']:
            if cut_queue is None:
                self.cut_queue = FastSolutionPool(len(self.rho_idx))
            else:
                assert isinstance(cut_queue, FastSolutionPool)
                self.cut_queue = cut_queue

        # setup pointer to polish_queue to send integer solutions to PolishAndRoundCallback
        if self.settings['polish_flag']:
            if polish_queue is None:
                self.polish_queue = FastSolutionPool(len(self.rho_idx))
            else:
                assert isinstance(polish_queue, FastSolutionPool)
                self.polish_queue = polish_queue

        # setup indices for update bounds
        if self.settings['chained_updates_flag']:
            self.loss_cut_constraint = [indices['loss'], [1.0]]
            if is_multiclass:
                self.mc_L0_cut_constraint = [[indices['mc_L0_norm']], [1.0]]
            else:
                self.L0_cut_constraint = [[indices['L0_norm']], [1.0]]
            self.objval_cut_constraint = [[indices['objval']], [1.0]]
            self.bound_cut_purge_flag = self.use_constraint.purge if self.settings[
                'purge_loss_cuts'] else self.use_constraint.force

        return

    def add_loss_cut(self, rho):

        loss_value, loss_slope = self.compute_loss_cut(rho)

        self.add(constraint=[self.cut_idx, [1.0] + (-loss_slope).tolist()],
                 sense="G",
                 rhs=float(loss_value - loss_slope.dot(rho)),
                 use=self.loss_cut_purge_flag)

        return loss_value

    def update_bounds(self):

        bounds = chained_updates(bounds=self.control['bounds'],
                                 C_0_nnz=self.C_0_nnz,
                                 new_objval_at_relaxation=self.control['lowerbound'],
                                 new_objval_at_feasible=self.control['upperbound'])

        # add cuts if bounds need to be tighter
        if bounds['loss_min'] > self.control['bounds']['loss_min']:
            self.add(constraint=self.loss_cut_constraint, sense="G", rhs=bounds['loss_min'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['loss_min'] = bounds['loss_min']
            self.control['n_bound_updates_loss_min'] += 1

        if bounds['objval_min'] > self.control['bounds']['objval_min']:
            self.add(constraint=self.objval_cut_constraint, sense="G", rhs=bounds['objval_min'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['objval_min'] = bounds['objval_min']
            self.control['n_bound_updates_objval_min'] += 1

        if bounds['L0_max'] < self.control['bounds']['L0_max']:
            self.add(constraint=self.L0_cut_constraint, sense="L", rhs=bounds['L0_max'], use=self.bound_cut_purge_flag)
            self.control['bounds']['L0_max'] = bounds['L0_max']
            self.control['n_bound_updates_L0_max'] += 1

        if bounds['loss_max'] < self.control['bounds']['loss_max']:
            self.add(constraint=self.loss_cut_constraint, sense="L", rhs=bounds['loss_max'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['loss_max'] = bounds['loss_max']
            self.control['n_bound_updates_loss_max'] += 1

        if bounds['objval_max'] < self.control['bounds']['objval_max']:
            self.add(constraint=self.objval_cut_constraint, sense="L", rhs=bounds['objval_max'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['objval_max'] = bounds['objval_max']
            self.control['n_bound_updates_objval_max'] += 1

        return

    def mc_update_bounds(self):

        bounds = mc_chained_updates(bounds=self.control['bounds'],
                                    mc_C_0_nnz=self.mc_C_0_nnz,
                                    new_objval_at_relaxation=self.control['lowerbound'],
                                    new_objval_at_feasible=self.control['upperbound'])

        # add cuts if bounds need to be tighter
        if bounds['loss_min'] > self.control['bounds']['loss_min']:
            self.add(constraint=self.loss_cut_constraint, sense="G", rhs=bounds['loss_min'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['loss_min'] = bounds['loss_min']
            self.control['n_bound_updates_loss_min'] += 1

        if bounds['objval_min'] > self.control['bounds']['objval_min']:
            self.add(constraint=self.objval_cut_constraint, sense="G", rhs=bounds['objval_min'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['objval_min'] = bounds['objval_min']
            self.control['n_bound_updates_objval_min'] += 1

        if bounds['mc_L0_max'] < self.control['bounds']['mc_L0_max']:
            self.add(constraint=self.mc_L0_cut_constraint, sense="L", rhs=bounds['mc_L0_max'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['mc_L0_max'] = bounds['mc_L0_max']
            self.control['n_bound_updates_L0_max'] += 1

        if bounds['loss_max'] < self.control['bounds']['loss_max']:
            self.add(constraint=self.loss_cut_constraint, sense="L", rhs=bounds['loss_max'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['loss_max'] = bounds['loss_max']
            self.control['n_bound_updates_loss_max'] += 1

        if bounds['objval_max'] < self.control['bounds']['objval_max']:
            self.add(constraint=self.objval_cut_constraint, sense="L", rhs=bounds['objval_max'],
                     use=self.bound_cut_purge_flag)
            self.control['bounds']['objval_max'] = bounds['objval_max']
            self.control['n_bound_updates_objval_max'] += 1

        return

    def __call__(self):

        # print_log('in cut callback')
        callback_start_time = time.time()

        # record entry metrics
        self.control['cut_callback_times_called'] += 1
        self.control['lowerbound'] = self.get_best_objective_value()
        self.control['relative_gap'] = self.get_MIP_relative_gap()
        self.control['nodes_processed'] = self.get_num_nodes()
        self.control['nodes_remaining'] = self.get_num_remaining_nodes()

        # add initial cuts first time the callback is used
        if self.initial_cuts is not None:
            print_log('adding %1.0f initial cuts' % len(self.initial_cuts['lhs']))
            for cut, lhs in zip(self.initial_cuts['coefs'], self.initial_cuts['lhs']):
                self.add(constraint=cut, sense="G", rhs=lhs, use=self.loss_cut_purge_flag)
            self.initial_cuts = None

        # get integer feasible solution
        rho = np.array(self.get_values(self.rho_idx))
        alpha = np.array(self.get_values(self.alpha_idx))
        beta = np.array(self.get_values(self.beta_idx))

        # check that CPLEX is actually integer. if not, then recast as int
        if not is_integer(rho):
            rho = cast_to_integer(rho)
            alpha = self.get_alpha(rho)
            beta = self.get_beta(rho)
        # add cutting plane at integer feasible solution
        cut_start_time = time.time()
        loss_value = self.add_loss_cut(rho)
        cut_time = time.time() - cut_start_time
        cuts_added = 1

        # if solution updates incumbent, then add solution to queue for polishing
        current_upperbound = float(
            loss_value + self.get_L0_penalty_from_alpha(alpha) + self.get_L0_penalty_from_beta(beta))
        incumbent_update = current_upperbound < self.control['upperbound']

        if incumbent_update:
            self.control['incumbent'] = rho
            self.control['upperbound'] = current_upperbound
            self.control['n_incumbent_updates'] += 1

        if self.settings['polish_flag']:
            polishing_cutoff = self.control['upperbound'] * (1.0 + self.settings['polishing_tolerance'])
            if current_upperbound < polishing_cutoff:
                self.polish_queue.add(current_upperbound, rho)

        # add cutting planes at other integer feasible solutions in cut_queue
        if self.settings['add_cuts_at_heuristic_solutions']:
            if len(self.cut_queue) > 0:
                self.cut_queue.filter_sort_unique()
                cut_start_time = time.time()
                for cut_rho in self.cut_queue.solutions:
                    self.add_loss_cut(cut_rho)
                cut_time += time.time() - cut_start_time
                cuts_added += len(self.cut_queue)
                self.cut_queue.clear()

        # update bounds
        if self.settings['chained_updates_flag']:
            if (self.control['lowerbound'] > self.control['bounds']['objval_min']) or (
                    self.control['upperbound'] < self.control['bounds']['objval_max']):
                self.control['n_update_bounds_calls'] += 1
                if self.is_multiclass:
                    self.mc_update_bounds()
                else:
                    self.update_bounds()

        # record metrics at end
        self.control['n_cuts'] += cuts_added
        self.control['total_cut_time'] += cut_time
        self.control['total_cut_callback_time'] += time.time() - callback_start_time
        # print_log('left cut callback')
        print(
            f"\rCuts: {self.control['n_cuts']}; Optimality Gap: {self.control['relative_gap']}; Loss value: {loss_value}; Nodes processed: {self.control['nodes_processed']}",
            end="")
        return


class PolishAndRoundCallback(HeuristicCallback):
    """
    This callback has to be initialized after construnction with initialize().

    HeuristicCallback is called intermittently during B&B by CPLEX. It runs several heuristics in a fast way and contains
    several options to stop early. Note: It is important for the callback to run quickly since it is called fairly often.
    If HeuristicCallback runs slowly, then it will slow down overall B&B progress.

    Heuristics include:

    - Runs sequential rounding on the continuous solution from the surrogate LP (only if there has been a change in the
      lower bound). Requires settings['round_flag'] = True. If settings['polish_after_rounding'] = True, then the
      rounded solutions are polished using DCD.

    - Polishes integer solutions in polish_queue using DCD. Requires settings['polish_flag'] = True.

    Optional:

    - Feasible solutions are passed to LazyCutConstraintCallback via cut_queue

    Known issues:

    - Sometimes CPLEX does not return an integer feasible solution (in which case we correct this manually)
    """

    def initialize(self, indices, control, settings, cut_queue, polish_queue, get_objval, get_L0_norm,
                   is_feasible, polishing_handle, rounding_handle, mc_get_L0_norm=None, is_multiclass=False):

        # todo: add basic assertions to make sure that nothing weird is going on
        assert isinstance(indices, dict)
        assert isinstance(control, dict)
        assert isinstance(settings, dict)
        assert isinstance(cut_queue, FastSolutionPool)
        assert isinstance(polish_queue, FastSolutionPool)
        assert callable(get_objval)
        assert callable(get_L0_norm)
        assert callable(is_feasible)
        assert callable(polishing_handle)
        assert callable(rounding_handle)

        self.rho_idx = indices['rho']
        self.L0_reg_ind = indices['L0_reg_ind']
        self.mc_L0_reg_ind = indices['mc_L0_reg_ind'] if 'mc_L0_reg_ind' in indices else []
        self.C_0_nnz = indices['C_0_nnz']
        self.indices = indices
        self.previous_lowerbound = 0.0
        self.control = control
        self.settings = settings

        self.round_flag = settings['round_flag']
        self.polish_rounded_solutions = settings['polish_rounded_solutions']
        self.polish_flag = settings['polish_flag']
        self.polish_queue = polish_queue

        self.cut_queue = cut_queue  # pointer to cut_queue

        self.rounding_tolerance = float(1.0 + settings['rounding_tolerance'])
        self.rounding_start_cuts = settings['rounding_start_cuts']
        self.rounding_stop_cuts = settings['rounding_stop_cuts']
        self.rounding_stop_gap = settings['rounding_stop_gap']
        self.rounding_start_gap = settings['rounding_start_gap']

        self.polishing_tolerance = float(1.0 + settings['polishing_tolerance'])
        self.polishing_start_cuts = settings['polishing_start_cuts']
        self.polishing_stop_cuts = settings['polishing_stop_cuts']
        self.polishing_stop_gap = settings['polishing_stop_gap']
        self.polishing_start_gap = settings['polishing_start_gap']
        self.polishing_max_solutions = settings['polishing_max_solutions']
        self.polishing_max_runtime = settings['polishing_max_runtime']

        self.get_objval = get_objval
        self.get_L0_norm = get_L0_norm
        self.mc_get_L0_norm = mc_get_L0_norm
        self.is_feasible = is_feasible
        self.polishing_handle = polishing_handle
        self.rounding_handle = rounding_handle
        self.is_multiclass = is_multiclass

        return

    def update_heuristic_flags(self, n_cuts, relative_gap):

        # keep on rounding?
        keep_rounding = (self.rounding_start_cuts <= n_cuts <= self.rounding_stop_cuts) and \
                        (self.rounding_stop_gap <= relative_gap <= self.rounding_start_gap)

        # keep on polishing?
        keep_polishing = (self.polishing_start_cuts <= n_cuts <= self.polishing_stop_cuts) and \
                         (self.polishing_stop_gap <= relative_gap <= self.polishing_start_gap)

        self.round_flag &= keep_rounding
        self.polish_flag &= keep_polishing
        self.polish_rounded_solutions &= self.round_flag

        return

    def __call__(self):
        # todo write rounding/polishing as separate function calls

        # print_log('in heuristic callback')
        if not (self.round_flag or self.polish_flag):
            return

        callback_start_time = time.time()
        self.control['heuristic_callback_times_called'] += 1
        self.control['upperbound'] = self.get_incumbent_objective_value()
        self.control['lowerbound'] = self.get_best_objective_value()
        self.control['relative_gap'] = self.get_MIP_relative_gap()

        # check if lower bound was updated since last call
        lowerbound_update = self.previous_lowerbound < self.control['lowerbound']
        if lowerbound_update:
            self.previous_lowerbound = self.control['lowerbound']

        # check if incumbent solution has been updated
        # if incumbent solution is not integer, then recast as integer and update objective value manually
        if self.has_incumbent():
            cplex_incumbent = np.array(self.get_incumbent_values(self.rho_idx))
            cplex_rounding_issue = not is_integer(cplex_incumbent)
            if cplex_rounding_issue:
                cplex_incumbent = cast_to_integer(cplex_incumbent)

            incumbent_update = not np.array_equal(cplex_incumbent, self.control['incumbent'])

            if incumbent_update:
                self.control['incumbent'] = cplex_incumbent
                self.control['n_incumbent_updates'] += 1
                if cplex_rounding_issue:
                    self.control['upperbound'] = self.get_objval(cplex_incumbent)

        # update flags on whether or not to keep rounding / polishing
        self.update_heuristic_flags(n_cuts=self.control['n_cuts'], relative_gap=self.control['relative_gap'])

        # variables to store best objective value / solution from heuristics
        best_objval = float('inf')
        best_solution = None

        # run sequential rounding if lower bound was updated since the last call
        if self.round_flag and lowerbound_update:

            rho_cts = np.array(self.get_values(self.rho_idx))
            zero_idx_rho_ceil = np.equal(np.ceil(rho_cts), 0)
            zero_idx_rho_floor = np.equal(np.floor(rho_cts), 0)
            cannot_round_to_zero = np.logical_not(np.logical_or(zero_idx_rho_ceil, zero_idx_rho_floor))
            min_l0_norm = np.count_nonzero(cannot_round_to_zero[self.L0_reg_ind])
            max_l0_norm = np.count_nonzero(rho_cts[self.L0_reg_ind])
            if self.is_multiclass:
                mc_min_l0_norm = \
                    np.sum(
                        (np.count_nonzero(cannot_round_to_zero.reshape(self.mc_L0_reg_ind.shape[0], -1), axis=1) > 0)[
                            self.mc_L0_reg_ind])
                mc_max_l0_norm = np.sum(
                    (np.count_nonzero(rho_cts.reshape(self.mc_L0_reg_ind.shape[0], -1), axis=1) > 0)[
                        self.mc_L0_reg_ind])
                rounded_solution_is_feasible = (
                        min_l0_norm < self.control['bounds']['L0_max'] and max_l0_norm > self.control['bounds'][
                    'L0_min'] and mc_min_l0_norm < self.control['bounds']['mc_L0_max'] and mc_max_l0_norm >
                        self.control['bounds']['mc_L0_min'])
            else:
                rounded_solution_is_feasible = (
                        min_l0_norm < self.control['bounds']['L0_max'] and max_l0_norm > self.control['bounds'][
                    'L0_min'])

            if rounded_solution_is_feasible:

                rounding_cutoff = self.rounding_tolerance * self.control['upperbound']
                rounding_start_time = time.time()
                rounded_solution, rounded_objval, early_stop = self.rounding_handle(rho_cts, rounding_cutoff)
                self.control['total_round_time'] += time.time() - rounding_start_time
                self.control['n_rounded'] += 1

                # round solution if sequential rounding did not stop early
                if not early_stop:

                    if self.settings['add_cuts_at_heuristic_solutions']:
                        self.cut_queue.add(rounded_objval, rounded_solution)

                    if self.is_multiclass:
                        if self.is_feasible(rounded_solution, L0_min=self.control['bounds']['L0_min'],
                                            L0_max=self.control['bounds']['L0_max'],
                                            mc_L0_min=self.control['bounds']['mc_L0_min'],
                                            mc_L0_max=self.control['bounds']['mc_L0_max']):
                            best_solution = rounded_solution
                            best_objval = rounded_objval
                    else:
                        if self.is_feasible(rounded_solution, L0_min=self.control['bounds']['L0_min'],
                                            L0_max=self.control['bounds']['L0_max']):
                            best_solution = rounded_solution
                            best_objval = rounded_objval

                    if self.polish_rounded_solutions:
                        current_upperbound = min(rounded_objval, self.control['upperbound'])
                        polishing_cutoff = current_upperbound * self.polishing_tolerance

                        if rounded_objval < polishing_cutoff:
                            start_time = time.time()
                            polished_solution, _, polished_objval = self.polishing_handle(rounded_solution)
                            self.control['total_round_then_polish_time'] += time.time() - start_time
                            self.control['n_rounded_then_polished'] += 1

                            if self.settings['add_cuts_at_heuristic_solutions']:
                                self.cut_queue.add(polished_objval, polished_solution)

                            if self.is_multiclass:
                                if self.is_feasible(polished_solution, L0_min=self.control['bounds']['L0_min'],
                                                    L0_max=self.control['bounds']['L0_max'],
                                                    mc_L0_min=self.control['bounds']['mc_L0_min'],
                                                    mc_L0_max=self.control['bounds']['mc_L0_max']):
                                    best_solution = polished_solution
                                    best_objval = polished_objval
                            else:
                                if self.is_feasible(polished_solution, L0_min=self.control['bounds']['L0_min'],
                                                    L0_max=self.control['bounds']['L0_max']):
                                    best_solution = polished_solution
                                    best_objval = polished_objval

        # polish solutions in polish_queue or that were produced by rounding
        if self.polish_flag and len(self.polish_queue) > 0:

            # get current upperbound
            current_upperbound = min(best_objval, self.control['upperbound'])
            polishing_cutoff = self.polishing_tolerance * current_upperbound
            self.polish_queue.filter_sort_unique(max_objval=polishing_cutoff)

            if len(self.polish_queue) > 0:
                polished_queue = FastSolutionPool(self.polish_queue.P)
                polish_time = 0
                n_polished = 0
                for objval, solution in zip(self.polish_queue.objvals, self.polish_queue.solutions):

                    if objval >= polishing_cutoff:
                        break

                    polish_start_time = time.time()
                    polished_solution, _, polished_objval = self.polishing_handle(solution)
                    polish_time += time.time() - polish_start_time
                    n_polished += 1

                    # update cutoff whenever the solution returns an infeasible solutions
                    if self.is_multiclass:
                        if self.is_feasible(polished_solution, L0_min=self.control['bounds']['L0_min'],
                                            L0_max=self.control['bounds']['L0_max'],
                                            mc_L0_min=self.control['bounds']['mc_L0_min'],
                                            mc_L0_max=self.control['bounds']['mc_L0_max']):
                            polished_queue.add(polished_objval, polished_solution)
                            current_upperbound = min(polished_objval, polished_objval)
                            polishing_cutoff = self.polishing_tolerance * current_upperbound
                    else:
                        if self.is_feasible(polished_solution, L0_min=self.control['bounds']['L0_min'],
                                            L0_max=self.control['bounds']['L0_max']):
                            polished_queue.add(polished_objval, polished_solution)
                            current_upperbound = min(polished_objval, polished_objval)
                            polishing_cutoff = self.polishing_tolerance * current_upperbound

                    if polish_time > self.polishing_max_runtime:
                        break

                    if n_polished > self.polishing_max_solutions:
                        break

                self.polish_queue.clear()
                self.control['total_polish_time'] += polish_time
                self.control['n_polished'] += n_polished

                if self.settings['add_cuts_at_heuristic_solutions']:
                    self.cut_queue.add(polished_queue.objvals, polished_queue.solutions)

                # check if the best polished solution will improve the queue
                polished_queue.filter_sort_unique(max_objval=best_objval)
                if len(polished_queue) > 0:
                    best_objval, best_solution = polished_queue.get_best_objval_and_solution()

        # if heuristics produces a better solution then update the incumbent
        heuristic_update = best_objval < self.control['upperbound']
        if heuristic_update:
            self.control['n_heuristic_updates'] += 1
            if self.is_multiclass:
                proposed_solution, proposed_objval = mc_convert_to_risk_slim_cplex_solution(indices=self.indices,
                                                                                            rho=best_solution,
                                                                                            objval=best_objval)
            else:
                proposed_solution, proposed_objval = convert_to_risk_slim_cplex_solution(indices=self.indices,
                                                                                         rho=best_solution,
                                                                                         objval=best_objval)
            self.set_solution(solution=proposed_solution, objective_value=proposed_objval)

        self.control['total_heuristic_callback_time'] += time.time() - callback_start_time
        # print_log('left heuristic callback')
        return


# DATA CONVERSION
def is_integer(x):
    """
    checks if numpy array is an integer vector

    Parameters
    ----------
    x

    Returns
    -------

    """
    return np.array_equal(x, np.require(x, dtype=np.int_))


def cast_to_integer(x):
    """
    casts numpy array to integer vector

    Parameters
    ----------
    x

    Returns
    -------

    """
    original_type = x.dtype
    return np.require(np.require(x, dtype=np.int_), dtype=original_type)
