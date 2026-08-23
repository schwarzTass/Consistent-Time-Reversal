import jax
import jax.numpy as jnp
import numpy as onp

from . import utils

def _canonical_reverse_indices_and_prefix_counts(counts,unique_values):
    """Return reverse indices and prefix totals for sparse canonical tuples."""
    minimum_state = jnp.min(unique_values)
    n_state_ids = jnp.max(unique_values) - minimum_state + 1
    shifted_values = (unique_values - minimum_state).astype(jnp.int32)

    tuple_codes = jnp.zeros(unique_values.shape[0],dtype=jnp.int32)
    reverse_codes = jnp.zeros(unique_values.shape[0],dtype=jnp.int32)
    for offset in range(unique_values.shape[1]):
        tuple_codes = tuple_codes*n_state_ids + shifted_values[:,offset]
        reverse_codes = (
            reverse_codes*n_state_ids + shifted_values[:,-offset-1]
        )

    # Sorting once serves two purposes: reverse tuples can be found with a
    # binary search, and tuples with a common prefix become contiguous.
    tuple_order = jnp.argsort(tuple_codes)
    sorted_codes = tuple_codes[tuple_order]
    reverse_positions = jnp.searchsorted(sorted_codes,reverse_codes,side="left")
    safe_reverse_positions = jnp.minimum(
        reverse_positions,unique_values.shape[0]-1
    )
    reverse_exists = (
        (reverse_positions < unique_values.shape[0])
        & (sorted_codes[safe_reverse_positions] == reverse_codes)
    )
    reverse_indices = tuple_order[safe_reverse_positions]

    prefix_codes = tuple_codes // n_state_ids
    sorted_prefix_codes = prefix_codes[tuple_order]
    starts_new_prefix = jnp.concatenate((
        jnp.ones(1,dtype=bool),
        sorted_prefix_codes[1:] != sorted_prefix_codes[:-1],
    ))
    prefix_group_indices = jnp.cumsum(
        starts_new_prefix,dtype=jnp.int32
    ) - 1
    prefix_group_counts = jnp.zeros(
        counts.shape,dtype=counts.dtype
    ).at[
        prefix_group_indices
    ].add(counts[tuple_order])
    sorted_prefix_counts = prefix_group_counts[prefix_group_indices]
    prefix_counts = jnp.zeros(
        counts.shape,dtype=counts.dtype
    ).at[tuple_order].set(
        sorted_prefix_counts
    )

    has_duplicate_tuple = jnp.any(sorted_codes[1:] == sorted_codes[:-1])
    return reverse_indices,reverse_exists,prefix_counts,has_duplicate_tuple


@jax.jit
def _dSk_est_jitted(counts,unique_values,avg_time_per_lump):
    (
        reverse_indices,
        reverse_exists,
        prefix_counts,
        has_duplicate_tuple,
    ) = _canonical_reverse_indices_and_prefix_counts(counts,unique_values)

    # The historical estimators allowed an all--1 padding tuple and omitted
    # only its entropy contribution. Canonical counting does not create this
    # tuple, but retaining the rule keeps this function backwards compatible.
    is_boundary_tuple = jnp.all(unique_values == -1,axis=1)
    active = (counts > 0) & ~is_boundary_tuple
    reverse_counts = counts[reverse_indices]
    missing_reverse = jnp.any(
        active & (~reverse_exists | (reverse_counts == 0))
    )

    float_dtype = (
        jnp.float64
        if counts.dtype == jnp.int64
        else jnp.result_type(avg_time_per_lump,jnp.float32)
    )
    total_count = jnp.sum(counts.astype(float_dtype))
    safe_total_count = jnp.where(total_count > 0,total_count,1)
    safe_counts = jnp.where(active,counts,1).astype(float_dtype)
    safe_reverse_counts = jnp.where(active,reverse_counts,1).astype(float_dtype)
    safe_prefix_counts = jnp.where(active,prefix_counts,1).astype(float_dtype)
    safe_reverse_prefix_counts = jnp.where(
        active,prefix_counts[reverse_indices],1
    ).astype(float_dtype)

    tuple_probabilities = safe_counts/safe_total_count
    forward_splitting_probabilities = safe_counts/safe_prefix_counts
    reverse_splitting_probabilities = (
        safe_reverse_counts/safe_reverse_prefix_counts
    )
    terms = tuple_probabilities*jnp.log(
        forward_splitting_probabilities/reverse_splitting_probabilities
    )
    value = jnp.sum(
        jnp.where(active,terms,jnp.zeros((),dtype=float_dtype))
    )/avg_time_per_lump
    return value,missing_reverse,total_count,has_duplicate_tuple


def _validate_canonical_estimator_inputs(
    counts,unique_values,k,avg_time_per_lump
):
    counts = jnp.asarray(counts)
    unique_values = jnp.asarray(unique_values)
    if counts.ndim != 1:
        raise ValueError(f"counts must be a 1D array. Got shape {counts.shape}")
    if unique_values.ndim != 2:
        raise ValueError(
            f"unique_values must be a 2D array. Got shape {unique_values.shape}"
        )
    if unique_values.shape[0] != counts.shape[0]:
        raise ValueError(
            "The number of canonical tuples must equal the number of counts. "
            f"Got {unique_values.shape[0]} tuples and {counts.shape[0]} counts"
        )
    if unique_values.shape[0] == 0:
        raise ValueError("unique_values must contain at least one tuple")
    if unique_values.shape[1] != k+1:
        raise ValueError(
            f"unique_values must have k+1={k+1} columns. "
            f"Got {unique_values.shape[1]}"
        )
    if k < 0:
        raise ValueError(f"k must be nonnegative. Got {k}")
    if not jnp.issubdtype(unique_values.dtype,jnp.integer):
        raise ValueError("unique_values must contain integer state ids")
    if bool(jax.device_get(jnp.any(counts < 0))):
        raise ValueError("counts must be nonnegative")

    avg_time = float(jax.device_get(jnp.asarray(avg_time_per_lump)))
    if not onp.isfinite(avg_time) or avg_time <= 0:
        raise ValueError(
            "avg_time_per_lump must be finite and greater than 0. "
            f"Got {avg_time_per_lump}"
        )

    minimum_state = int(jax.device_get(jnp.min(unique_values)))
    maximum_state = int(jax.device_get(jnp.max(unique_values)))
    n_state_ids = maximum_state - minimum_state + 1
    if n_state_ids**(k+1)-1 > onp.iinfo(onp.int32).max:
        raise ValueError("Encoded canonical tuples do not fit in int32")
    return counts,unique_values



def dSk_est(k_plus_1_counts_over_all_paths,k_plus_1_unique_values,k,avg_time_per_lump):
    """
    Input:
        k_plus_1_counts_over_all_paths: array of shape (n_unique_tuples,). It contains the counts of each of the k+1 tuples.
        k_plus_1_unique_values: array of shape (n_unique_tuples,k+1). It contains the unique values of the k+1 tuples. I.e. k_plus_1_counts[i] is the count of the k+1 tuple k_plus_1_unique_values[i].
        k: integer, the order of the estimator.
        avg_time_per_lump: float, the average time per lump.
    Output:
        dSk_est: float, the estimated dSk.
    """

    counts,unique_values = _validate_canonical_estimator_inputs(
        k_plus_1_counts_over_all_paths,
        k_plus_1_unique_values,
        k,
        avg_time_per_lump,
    )
    value,missing_reverse,total_count,has_duplicate_tuple = _dSk_est_jitted(
        counts,unique_values,avg_time_per_lump
    )
    if bool(jax.device_get(total_count == 0)):
        raise ValueError("counts must contain at least one occurrence")
    if bool(jax.device_get(has_duplicate_tuple)):
        raise ValueError("unique_values must not contain duplicate tuples")
    if bool(jax.device_get(missing_reverse)):
        raise ValueError(
            "A tuple with positive count has an absent or zero-count reverse "
            "tuple. Simulate longer or check time reversibility."
        )
    return value

def dSk_est_experimentalist(paths_meso_states,paths_meso_times,k):
    """
    Input:
        paths_meso_states: array of shape (N_paths,N_steps). The mesoscopic states of the paths.
        paths_meso_times: array of shape (N_paths,N_steps). The mesoscopic times of the paths.
        k: integer, the order of the estimator.
    Output:
        dSk_est: float, the estimated dSk.
    """
    assert len(paths_meso_states.shape) == 2, f"paths_meso_states must be a 2D array. It is a {len(paths_meso_states.shape)}D array."
    assert len(paths_meso_times.shape) == 2, f"paths_meso_times must be a 2D array. It is a {len(paths_meso_times.shape)}D array."
    assert paths_meso_states.shape[0] == paths_meso_times.shape[0], f"The number of paths must be equal to the number of time steps. {paths_meso_states.shape[0]} != {paths_meso_times.shape[0]}"
    assert k >= 1, f"k must be greater than or equal to 1. {k} < 1"
    assert k <= 10, f"k must be at most 10. {k} > 10"

    # we first get the counts
    k_plus_1_tuples = utils.paths_to_k_th_order(paths_meso_states,k+1)
    k_plus_1_counts,k_plus_1_unique_values,count_success = utils.count_occurances(k_plus_1_tuples)
    assert count_success, f"For order k={k}, count_occurances was unsucessfull. That means, either there are not enough steps recorded (increase max_steps_per_path [if you run into memory constraints then, decrease N_paths_per_batch]), or remove {k} from estimators_to_use."
    k_plus_1_counts_over_all_paths = jnp.sum(k_plus_1_counts,axis=0)
    avg_time_per_lump = jnp.mean(paths_meso_times,where=paths_meso_states!=-1)
    return dSk_est(k_plus_1_counts_over_all_paths,k_plus_1_unique_values,k,avg_time_per_lump)

def unconditioned_estimator(k_plus_1_counts_over_all_paths,k_plus_1_unique_values,k,avg_time_per_lump,harunari_version = False):
    """
    Input:
        k_plus_1_counts_over_all_paths: array of shape (n_unique_tuples,). It contains the counts of each of the k+1 tuples.
        k_plus_1_unique_values: array of shape (n_unique_tuples,k+1). It contains the unique values of the k+1 tuples. I.e. k_plus_1_counts[i] is the count of the k+1 tuple k_plus_1_unique_values[i].
        k: integer, the order of the estimator.
        avg_time_per_lump: float, the average time per lump.
    Output:
        the unconditioned estimator value. If harunari_version is True, then divide by 2.
    """

    counts,unique_values = _validate_canonical_estimator_inputs(
        k_plus_1_counts_over_all_paths,
        k_plus_1_unique_values,
        k,
        avg_time_per_lump,
    )
    value,missing_reverse,total_count,has_duplicate_tuple = (
        _unconditioned_estimator_jitted(
            counts,unique_values,avg_time_per_lump
        )
    )
    if bool(jax.device_get(total_count == 0)):
        raise ValueError("counts must contain at least one occurrence")
    if bool(jax.device_get(has_duplicate_tuple)):
        raise ValueError("unique_values must not contain duplicate tuples")
    if bool(jax.device_get(missing_reverse)):
        raise ValueError(
            "A tuple with positive count has an absent or zero-count reverse "
            "tuple. Simulate longer or check time reversibility."
        )
    if harunari_version:
        value = value/2
    return value


@jax.jit
def _unconditioned_estimator_jitted(
    counts,unique_values,avg_time_per_lump
):
    (
        reverse_indices,
        reverse_exists,
        _,
        has_duplicate_tuple,
    ) = _canonical_reverse_indices_and_prefix_counts(counts,unique_values)
    is_boundary_tuple = jnp.all(unique_values == -1,axis=1)
    active = (counts > 0) & ~is_boundary_tuple
    reverse_counts = counts[reverse_indices]
    missing_reverse = jnp.any(
        active & (~reverse_exists | (reverse_counts == 0))
    )

    float_dtype = (
        jnp.float64
        if counts.dtype == jnp.int64
        else jnp.result_type(avg_time_per_lump,jnp.float32)
    )
    total_count = jnp.sum(counts.astype(float_dtype))
    safe_total_count = jnp.where(total_count > 0,total_count,1)
    safe_counts = jnp.where(active,counts,1).astype(float_dtype)
    safe_reverse_counts = jnp.where(active,reverse_counts,1).astype(float_dtype)
    tuple_probabilities = safe_counts/safe_total_count
    terms = tuple_probabilities*jnp.log(safe_counts/safe_reverse_counts)
    value = jnp.sum(
        jnp.where(active,terms,jnp.zeros((),dtype=float_dtype))
    )/avg_time_per_lump
    return value,missing_reverse,total_count,has_duplicate_tuple
