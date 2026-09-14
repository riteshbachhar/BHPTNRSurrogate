##==============================================================================
## BHPTNRSurrogate module
## Description : evaluates GPR surrogate fits at the EIM nodes, vectorized over nodes
## Author: Ritesh Bachhar, Sep 2026
##
##
## NOTE: despite this file being in the common_utils directory, it is specific
## to the BHPTNRSur2dq1e3 model and its associated h5 data file.
##
##
## The fits stored in the h5 file are sklearn GaussianProcessRegressor fits, and
## eval_pysur/evaluate_fit.py evaluates them by rebuilding the sklearn object and
## calling predict, one EIM node at a time. Its prediction for a test point x is
##
##     y_normed = kernel_(x, X_train_) . alpha_
##
## and for the kernel these fits use, Sum(Product(ConstantKernel, RBF), WhiteKernel),
## the WhiteKernel term is nonzero only where the two points coincide. A test point
## is never a training point, so it drops out exactly and each node reduces to a
## closed form -- see evaluate_stacked below.
##
## The hyperparameters (constant_value, length_scale) and the dual coefficients
## (alpha_) are fitted per node, but X_train_ is shared by every node of every mode,
## so once the per-node data is stacked the whole datapiece is one broadcast.
##
## Assembling the stacked arrays and evaluating them are kept separate on purpose:
## the algebra is the same wherever it runs, but where the assembly happens decides
## whether it is paid once or on every call. See build_stacked_fit.
##==============================================================================

import numpy as np

#----------------------------------------------------------------------------------------------------
def check_fit_is_stackable(node_index, node):
    """ The closed form in evaluate_stacked is only valid for the kernel these fits
        were actually built with. Fail loudly rather than quietly evaluate the wrong
        thing if that ever stops being true.

    Inputs
    ======

        node_index : index of the EIM node, used only in the error message

        node : dictionary of stored fit data for one EIM node, as read from the h5
               file by load_GPRs.py
    """

    kernel = node['GPR_params']['kernel_']
    expected = [(node['fitType'], 'GPR'), (kernel['name'], 'Sum'),
                (kernel['k1']['name'], 'Product'), (kernel['k2']['name'], 'WhiteKernel'),
                (kernel['k1']['k1']['name'], 'ConstantKernel'), (kernel['k1']['k2']['name'], 'RBF')]
    for found, want in expected:
        if found != want:
            raise ValueError("node %d: expected %r in the fit structure, found %r; the "
                             "closed form in eval_GPRs.py does not apply"
                             % (node_index, want, found))

    # evaluate_fit.py falls back to _y_train_std=1 when the key is absent, which is the
    # case for this model; both of these are folded into the algebra in evaluate_stacked
    if '_y_train_std' in node['GPR_params']:
        raise ValueError("node %d: fit carries _y_train_std, which the closed form in "
                         "eval_GPRs.py assumes is 1" % node_index)
    if np.any(np.asarray(node['GPR_params']['_y_train_mean']) != 0.0):
        raise ValueError("node %d: fit carries a nonzero _y_train_mean, which the closed "
                         "form in eval_GPRs.py assumes is 0" % node_index)


#----------------------------------------------------------------------------------------------------
def build_stacked_fit(nodes):
    """ Stack the stored fit data of every EIM node of one datapiece into contiguous
        arrays that evaluate_stacked can consume in a single broadcast.

        None of this depends on the evaluation point, so it only ever needs doing once
        per datapiece: BHPTNRSur2dq1e3 does it when the h5 file is loaded (see
        build_stacked_fits), and caches the result alongside the raw fit data.

    Inputs
    ======

        nodes : list of the per-node fit dictionaries of one datapiece of one mode,
                i.e. [h_eim_gpr_mode['node0'], h_eim_gpr_mode['node1'], ...]

    Outputs
    =======

        dictionary of stacked arrays, keyed as expected by evaluate_stacked
    """

    for i, node in enumerate(nodes):
        check_fit_is_stackable(i, node)

    # X_train_ is common to every node; only the hyperparameters and alpha_ differ
    X_train = np.asarray(nodes[0]['GPR_params']['X_train_'], dtype=float)
    for i, node in enumerate(nodes):
        if not np.array_equal(node['GPR_params']['X_train_'], X_train):
            raise ValueError("node %d: X_train_ differs from node 0; the stacking in "
                             "eval_GPRs.py assumes one shared training set" % i)

    # (n_nodes, 2) : one anisotropic length scale per node, one entry per parameter
    length_scale = np.array([node['GPR_params']['kernel_']['k1']['k2']['length_scale']
                             for node in nodes], dtype=float)
    inv_length_scale = 1.0/length_scale

    return {
        # pre-divide the shared training set by each node's length scale, so that the
        # per-call work is a subtraction rather than a division
        'X_train_scaled'   : np.ascontiguousarray(X_train[None,:,:]*inv_length_scale[:,None,:]),
        'inv_length_scale' : np.ascontiguousarray(inv_length_scale),
        # (n_nodes,) : overall variance of the RBF, per node
        'constant_value'   : np.array([node['GPR_params']['kernel_']['k1']['k1']['constant_value']
                                       for node in nodes], dtype=float).reshape(-1),
        # (n_nodes, n_train) : dual coefficients, K_train^-1 . y_train, solved at fit time
        'alpha_'           : np.ascontiguousarray([node['GPR_params']['alpha_'] for node in nodes],
                                                  dtype=float),
        # the fits were made on standardized, linearly detrended data
        'data_mean'        : np.array([node['data_mean'] for node in nodes], dtype=float).reshape(-1),
        'data_std'         : np.array([node['data_std'] for node in nodes], dtype=float).reshape(-1),
        'coef_'            : np.array([node['lin_reg_params']['coef_'] for node in nodes], dtype=float),
        'intercept_'       : np.array([node['lin_reg_params']['intercept_'] for node in nodes],
                                      dtype=float).reshape(-1),
    }


#----------------------------------------------------------------------------------------------------
def build_stacked_fits(fit_data_dict):
    """ Stack every mode of one datapiece, by calling build_stacked_fit per mode.

    Inputs
    ======

        fit_data_dict : dictionary of raw GPR fit data for one datapiece, as read from
                        the h5 file by load_GPRs.py. Keys are the modes; each value is
                        the pair [h_eim_gpr_mode, eim_indicies].

    Outputs
    =======

        dictionary of stacked arrays, one per mode, keyed by mode. Each value is what
        evaluate_stacked (and so fits._evaluate_GPR_at_EIM_nodes) expects as fit data.
    """

    stacked = {}
    for mode, (h_eim_gpr_mode, eim_indicies) in fit_data_dict.items():
        nodes = [h_eim_gpr_mode['node%s'%i] for i in range(len(eim_indicies))]
        stacked[mode] = build_stacked_fit(nodes)
    return stacked


#----------------------------------------------------------------------------------------------------
def evaluate_stacked(X, stacked_fit):
    """ Evaluate every EIM node of one datapiece at a single parameter point.

        Per node, this computes

            K = constant_value * exp(-0.5 * ||(x - X_train_)/length_scale||^2)
            y = (K . alpha_) * data_std + data_mean + coef_ . x + intercept_

        which is sklearn's prediction with the WhiteKernel term dropped (exactly zero
        away from the training points) and the normalization folded in analytically.

    Inputs
    ======

        X : parameter point, e.g. [log10(q), chi]

        stacked_fit : dictionary of stacked arrays from build_stacked_fit

    Outputs
    =======

        array of the fit value at each EIM node, shape (n_nodes,)
    """

    x = np.asarray(X, dtype=float)

    # (n_nodes, n_train, 2) : separation from each training point, in units of that
    # node's own length scale
    diff = x*stacked_fit['inv_length_scale'][:,None,:] - stacked_fit['X_train_scaled']
    K = stacked_fit['constant_value'][:,None] * np.exp(-0.5*np.sum(diff*diff, axis=-1))

    # np.sum(..., axis=1) and not einsum: this contraction cancels over ~8 orders of
    # magnitude, and pairwise summation is worth ~3x in accuracy here
    y = np.sum(K*stacked_fit['alpha_'], axis=1)

    # undo the standardization, then add back the linear trend that was fitted out
    return y*stacked_fit['data_std'] + stacked_fit['data_mean'] \
           + stacked_fit['coef_'] @ x + stacked_fit['intercept_']
