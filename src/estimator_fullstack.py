from functools import partial
import time

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jr
import numpy as onp

from . import coarse_graining as cg
from . import estimators as ests
from . import simulate


def _get_gpu_zero():
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError as error:
        raise RuntimeError("estimate_online requires GPU 0, but JAX found no GPU") from error
    if len(gpu_devices) == 0:
        raise RuntimeError("estimate_online requires GPU 0, but JAX found no GPU")
    return gpu_devices[0]


_HASH_MULTIPLIER = onp.uint32(2654435761)


def _get_count_dtype(maximum_count):
    """Choose the smallest safe count dtype supported by this JAX process."""
    if maximum_count <= onp.iinfo(onp.int32).max:
        return jnp.int32
    if maximum_count > onp.iinfo(onp.int64).max:
        raise ValueError(
            f"The requested count {maximum_count} exceeds int64 capacity"
        )
    if not jax.config.x64_enabled:
        raise ValueError(
            f"The requested count {maximum_count} exceeds int32 capacity, "
            "but JAX x64 is disabled. Set JAX_ENABLE_X64=True before importing "
            "JAX, restart the notebook kernel, and verify "
            "jax.config.x64_enabled."
        )
    return jnp.int64


def _create_canonical_index_lookup(canonical_tuples,n_states):
    """Build an O(U)-memory hash table from tuple codes to canonical indices."""
    canonical_tuples = onp.asarray(jax.device_get(canonical_tuples),dtype=onp.int64)
    if canonical_tuples.ndim != 2 or canonical_tuples.shape[0] == 0:
        raise ValueError(
            "canonical_tuples must be a nonempty two-dimensional array"
        )
    tuple_length = canonical_tuples.shape[1]
    canonical_codes = onp.zeros(canonical_tuples.shape[0],dtype=onp.int64)
    for offset in range(tuple_length):
        canonical_codes = canonical_codes*n_states + canonical_tuples[:,offset]
    if onp.any(canonical_codes > onp.iinfo(onp.int32).max):
        raise ValueError("Encoded tuples do not fit in int32")
    if onp.unique(canonical_codes).shape[0] != canonical_codes.shape[0]:
        raise ValueError("canonical_tuples must not contain duplicates")

    lookup_size = 1
    while lookup_size < 2*canonical_codes.shape[0]:
        lookup_size *= 2
    lookup_keys = onp.full(lookup_size,-1,dtype=onp.int32)
    lookup_indices = onp.full(lookup_size,-1,dtype=onp.int32)
    lookup_mask = lookup_size - 1
    for canonical_index,canonical_code in enumerate(canonical_codes):
        lookup_position = (
            int(canonical_code)*int(_HASH_MULTIPLIER)
        ) & lookup_mask
        while lookup_keys[lookup_position] != -1:
            lookup_position = (lookup_position+1) & lookup_mask
        lookup_keys[lookup_position] = canonical_code
        lookup_indices[lookup_position] = canonical_index
    return jnp.asarray(lookup_keys),jnp.asarray(lookup_indices)


def _encode_windows_block(states_block,n_states,tuple_length):
    n_windows = states_block.shape[1] - tuple_length + 1
    codes = jnp.zeros((states_block.shape[0],n_windows),dtype=jnp.int32)
    valid_windows = jnp.ones((states_block.shape[0],n_windows),dtype=bool)
    for offset in range(tuple_length):
        states = states_block[:,offset:offset+n_windows]
        valid_windows = valid_windows & (states >= 0)
        codes = codes*n_states + jnp.maximum(states,0)
    return codes,valid_windows


def _count_encoded_block_lookup(
    states_block,
    lookup_keys,
    lookup_indices,
    n_canonical_tuples,
    n_states,
    tuple_length,
    block_count_dtype,
):
    codes,valid_windows = _encode_windows_block(
        states_block,n_states,tuple_length
    )
    flat_codes = codes.reshape(-1)
    valid_windows = valid_windows.reshape(-1)
    lookup_mask = lookup_keys.shape[0] - 1
    lookup_positions = (
        flat_codes.astype(jnp.uint32)*jnp.uint32(_HASH_MULTIPLIER)
    ) & jnp.uint32(lookup_mask)
    lookup_positions = lookup_positions.astype(jnp.int32)
    canonical_indices = jnp.full(flat_codes.shape,-1,dtype=jnp.int32)
    unresolved = valid_windows

    def some_windows_are_unresolved(carry):
        _,_,unresolved = carry
        return jnp.any(unresolved)

    def probe_next_lookup_position(carry):
        lookup_positions,canonical_indices,unresolved = carry
        keys_at_positions = lookup_keys[lookup_positions]
        found = unresolved & (keys_at_positions == flat_codes)
        absent = unresolved & (keys_at_positions == -1)
        canonical_indices = jnp.where(
            found,lookup_indices[lookup_positions],canonical_indices
        )
        unresolved = unresolved & ~found & ~absent
        lookup_positions = jnp.where(
            unresolved,(lookup_positions+1) & lookup_mask,lookup_positions
        )
        return lookup_positions,canonical_indices,unresolved

    _,canonical_indices,_ = lax.while_loop(
        some_windows_are_unresolved,
        probe_next_lookup_position,
        (lookup_positions,canonical_indices,unresolved),
    )
    known_windows = valid_windows & (canonical_indices >= 0)
    block_counts = jnp.bincount(
        jnp.maximum(canonical_indices,0),
        weights=known_windows.astype(block_count_dtype),
        length=n_canonical_tuples,
    )
    unknown_count = jnp.sum(
        valid_windows & ~known_windows,dtype=block_count_dtype
    )
    return block_counts,unknown_count


@partial(
    jax.jit,
    static_argnames=(
        "n_canonical_tuples",
        "n_states",
        "tuple_length",
        "chunk_size",
        "count_dtype",
        "block_count_dtype",
    ),
)
def _count_canonical_tuples_lookup_jitted(
    paths_states,
    lookup_keys,
    lookup_indices,
    n_canonical_tuples,
    *,
    n_states,
    tuple_length,
    chunk_size,
    count_dtype,
    block_count_dtype,
):
    """Map every code directly to its canonical output-array index."""
    n_paths,n_steps = paths_states.shape
    n_windows = n_steps - tuple_length + 1
    n_full_chunks = n_windows//chunk_size
    n_remaining_windows = n_windows%chunk_size
    counts = jnp.zeros(n_canonical_tuples,dtype=count_dtype)
    unknown_count = jnp.zeros((),dtype=count_dtype)

    def count_full_chunk(chunk_index,carry):
        counts,unknown_count = carry
        states_block = lax.dynamic_slice(
            paths_states,
            (0,chunk_index*chunk_size),
            (n_paths,chunk_size+tuple_length-1),
        )
        block_counts,block_unknown_count = _count_encoded_block_lookup(
            states_block,
            lookup_keys,
            lookup_indices,
            n_canonical_tuples,
            n_states,
            tuple_length,
            block_count_dtype,
        )
        return (
            counts+block_counts.astype(count_dtype),
            unknown_count+block_unknown_count.astype(count_dtype),
        )

    counts,unknown_count = lax.fori_loop(
        0,n_full_chunks,count_full_chunk,(counts,unknown_count)
    )
    if n_remaining_windows:
        states_block = lax.dynamic_slice(
            paths_states,
            (0,n_full_chunks*chunk_size),
            (n_paths,n_remaining_windows+tuple_length-1),
        )
        block_counts,block_unknown_count = _count_encoded_block_lookup(
            states_block,
            lookup_keys,
            lookup_indices,
            n_canonical_tuples,
            n_states,
            tuple_length,
            block_count_dtype,
        )
        counts = counts + block_counts.astype(count_dtype)
        unknown_count = unknown_count + block_unknown_count.astype(count_dtype)
    return counts,unknown_count


def _count_canonical_tuples_with_lookup(
    paths_states,
    lookup_keys,
    lookup_indices,
    n_canonical_tuples,
    n_states,
    tuple_length,
    chunk_size,
    count_dtype,
):
    paths_states = jnp.asarray(paths_states,dtype=jnp.int32)
    if paths_states.shape[1] < tuple_length:
        return jnp.zeros(n_canonical_tuples,dtype=count_dtype)
    effective_chunk_size = min(
        chunk_size,paths_states.shape[1]-tuple_length+1
    )
    block_count_dtype = _get_count_dtype(
        paths_states.shape[0]*effective_chunk_size
    )
    counts,unknown_count = _count_canonical_tuples_lookup_jitted(
        paths_states,
        lookup_keys,
        lookup_indices,
        n_canonical_tuples,
        n_states=n_states,
        tuple_length=tuple_length,
        chunk_size=effective_chunk_size,
        count_dtype=count_dtype,
        block_count_dtype=block_count_dtype,
    )
    if unknown_count != 0:
        raise ValueError(
            f"Observed {int(unknown_count)} valid tuples that are absent from canonical_tuples"
        )
    return counts


def count_canonical_tuples(
    paths_states,
    canonical_tuples,
    n_states,
    chunk_size=1_000_000,
):
    """
    Count path windows with a direct canonical-index lookup.

    Every tuple is encoded as a base-n_states integer. An O(U)-memory hash
    table maps that code directly to the tuple's index in canonical_tuples,
    so the returned counts have exactly the same ordering. Only chunk_size
    starting positions are encoded at once.
    """
    paths_states = jnp.asarray(paths_states,dtype=jnp.int32)
    canonical_tuples = jnp.asarray(canonical_tuples,dtype=jnp.int32)
    if paths_states.ndim != 2:
        raise ValueError(f"paths_states must be 2D. Got shape {paths_states.shape}")
    if canonical_tuples.ndim != 2:
        raise ValueError(f"canonical_tuples must be 2D. Got shape {canonical_tuples.shape}")
    if canonical_tuples.shape[0] == 0:
        raise ValueError("canonical_tuples must contain at least one tuple")
    if n_states < 1:
        raise ValueError(f"n_states must be at least 1. Got {n_states}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1. Got {chunk_size}")
    tuple_length = canonical_tuples.shape[1]
    if tuple_length < 1:
        raise ValueError("canonical tuples must contain at least one state")
    maximum_count = paths_states.shape[0]*max(
        paths_states.shape[1]-tuple_length+1,0
    )
    count_dtype = _get_count_dtype(maximum_count)
    if n_states**tuple_length - 1 > onp.iinfo(onp.int32).max:
        raise ValueError("Encoded tuples do not fit in int32")
    if not jnp.all(
        (canonical_tuples >= 0) & (canonical_tuples < n_states)
    ):
        raise ValueError(f"canonical_tuples must contain state ids in 0...{n_states-1}")

    lookup_keys,lookup_indices = _create_canonical_index_lookup(
        canonical_tuples,n_states
    )

    return _count_canonical_tuples_with_lookup(
        paths_states,
        lookup_keys,
        lookup_indices,
        canonical_tuples.shape[0],
        n_states,
        tuple_length,
        chunk_size,
        count_dtype,
    )


@partial(jax.jit,static_argnames=("count_dtype",))
def _sum_lump_times_and_count(states,times,*,count_dtype=jnp.int32):
    """Return total dwell time and number of valid lumps in one batch."""
    valid_lumps = states != -1
    time_dtype = jnp.float64 if count_dtype == jnp.int64 else jnp.float32
    total_time = jnp.sum(
        jnp.where(valid_lumps,times,0),dtype=time_dtype
    )
    number_of_lumps = jnp.sum(valid_lumps,dtype=count_dtype)
    return total_time,number_of_lumps



def get_list_of_possible_transitions(transition_prob_array,transition_length):
    """
    transition_prob_array: jnp array of shape (n_states,n_states). On off-diagonals, it contains at transition_prob_array[i,j] the probability of transitioning from state i to state j.
    transition_length: int, the length of the transitions to consider. For example, if transition_length=2, we will consider all possible transitions of length 2, i.e. all possible pairs of states (i,j) such that there is a nonzero probability of transitioning from state i to state j. If k=3, we would determine all (i,j,k) so that there is a nonezro probability of transitioning from state i to state j and then from state j to state k.

    Returns:
        list of tuples, where each tuple is a possible (probability nonzero)transition of length transition_length.
    """
    # The graph is small and fixed while this Python list is constructed.
    # Copying it once avoids a device synchronization for every candidate edge.
    transition_prob_array = onp.asarray(jax.device_get(transition_prob_array))
    n_states = transition_prob_array.shape[0]
    # we first determine the possible transitions of length 1
    possible_transitions = [(state,) for state in range(n_states)]

    # we extend every possible transition by one state at a time. the diagonal is ignored because it contains exit rates rather than transitionprobabilities.
    for _ in range(1,transition_length):
        possible_transitions = [
            transition + (next_state,)
            for transition in possible_transitions
            for next_state in range(n_states)
            if next_state != transition[-1]
            and transition_prob_array[transition[-1],next_state] != 0
        ]

    return possible_transitions




def estimate_online(
    transition_prob_array,
    micro_to_meso_array,
    N_batches,
    N_paths_per_batch,
    steps_warmup,
    max_steps_per_path,
    estimators_to_use,
    key,
    *,
    counting_chunk_size=1_000_000,
):
    """
    EPR estimation and simulation as an online algorithm.

    transition_prob_array: array of shape (n_states,n_states). On off-diagonals, it contains at transition_prob_array[i,j] the probability of transitioning from state i to state j. Diagonal transition_prob_array[i,i] contains the exit rates of the state i.
    micro_to_meso_array: array of shape (n_states,). It contains at index i the meso state number of the micro state i.
    N_batches: int, the number of batches to use for the estimation. For fastest simulation, this should be 1. However, if there is a memory issue, this can be increased while decreasing the number of paths per batch. The total number of paths is N_batches*N_paths_per_batch.
    N_paths_per_batch: int, the number of paths to simulate for each batch.
    steps_warmup: int, the number of steps to warmup the system. During warmup, we do not record the states.
    max_steps_per_path: int, the maximum number of steps to simulate for each path. Thus, the statistics is done on max_steps-steps_warmup many steps
    estimators_to_use: list of ints, which are the orders of the estimators to use for the estimation.
        e.g. estimators_to_use = [1,2,3] will use the first order estimator, the second order estimator, and the third order estimator.
        special cases: -k will give both Skest and the unconditioned estimator for order k.
        e.g. estimators_to_use = [-1,-2,3,-4] will determine the S1est, S1uncondest, S2est, S2ucondest, S3est, S4est, S4uncondest
    key: jr.PRNGKey, the random key to use for the estimation.
    counting_chunk_size: int, the number of tuple starting positions per path processed in one counting iteration. A larger value reduces loop overhead but uses more temporary GPU memory; a smaller value reduces memory usage but may make counting slower. Chunk boundaries overlap by the estimator order, so every tuple is counted exactly once, including tuples crossing a boundary. This parameter does not change the resulting counts. The default is 1,000,000.
    Returns:
        Dictionary containing the microscopic entropy production rate under
        "dS_0" and the conditioned order-k estimate under "dS_k" for every
        requested k. A negative requested order additionally produces the
        unconditioned estimate under "dS_k,uncond".
    """

    counting_device = _get_gpu_zero()
    transition_prob_array = jax.device_put(transition_prob_array,counting_device)
    micro_to_meso_array = jax.device_put(micro_to_meso_array,counting_device)
    key = jax.device_put(key,counting_device)
    n_micro_states = transition_prob_array.shape[0]
    n_meso_states = int(jax.device_get(jnp.max(micro_to_meso_array))) + 1

    # we check if the input is valid
    # number of steps must be strictly larger than the number of warmup steps
    if N_batches < 1:
        raise ValueError(f"N_batches must be at least 1. Got {N_batches}")
    if N_paths_per_batch < 1:
        raise ValueError(
            f"N_paths_per_batch must be at least 1. Got {N_paths_per_batch}"
        )
    if max_steps_per_path <= steps_warmup:
        raise ValueError(f"max_steps_per_path must be strictly larger than steps_warmup. Got max_steps_per_path={max_steps_per_path}, steps_warmup={steps_warmup}")
    if counting_chunk_size < 1:
        raise ValueError(f"counting_chunk_size must be at least 1. Got {counting_chunk_size}")
    maximum_total_count = (
        N_batches
        * N_paths_per_batch
        * (max_steps_per_path-steps_warmup)
    )
    count_dtype = _get_count_dtype(maximum_total_count)
    time_dtype = jnp.float64 if count_dtype == jnp.int64 else jnp.float32
    print(f"Simulating {N_batches} batches of {N_paths_per_batch} paths each.\nPer path, we will simulate {max_steps_per_path} steps, of which the first {steps_warmup} steps are warmup steps.\nThus, we will record {max_steps_per_path-steps_warmup} steps per path, which will be used for the estimation of the EPR.")
    print(f"Thus, effective number of steps for the estimation: {maximum_total_count}. In base of 10, this is 10^{jnp.log10(maximum_total_count):.2f}")
    print(
        f"Counting independently on {counting_device} with canonical-index "
        f"lookup, count dtype {jnp.dtype(count_dtype).name}, and "
        f"chunk_size={counting_chunk_size}."
    )

    # we check that the mesostates are indeed a sequence 0...n_meso_states-1
    if not jnp.array_equal(jnp.sort(jnp.unique(micro_to_meso_array)), jnp.arange(n_meso_states)):
        raise ValueError(f"Mesostates must be a sequence 0...{n_meso_states-1}. Got {jnp.sort(jnp.unique(micro_to_meso_array))}")
    
    # check validity of estimator order: all must be integers, and nonzero
    for estimator_order in estimators_to_use:
        if not isinstance(estimator_order,int):
            raise ValueError(f"Estimator order must be an integer. Got {estimator_order} of type {type(estimator_order)}")
        if estimator_order == 0:
            raise ValueError(f"Estimator order must be nonzero. Got {estimator_order}")
    estimator_order_abs_values = [abs(estimator_order) for estimator_order in estimators_to_use]
    # we check if there are duplicates in estimator_order_abs_values
    if len(estimator_order_abs_values) != len(set(estimator_order_abs_values)):
        duplicates = set([x for x in estimator_order_abs_values if estimator_order_abs_values.count(x) > 1])
        raise ValueError(f"Estimator orders must be unique in absolute value. Recall: If you want Skest, add k. If you want BOTH Skest and Skundcondest,  add -k, but not k. Got duplicates: {duplicates}")
    
    # get the simulator function
    simulator = simulate.get_simulator_batched(transition_prob_array,steps_warmup,max_steps_per_path)



    # Note that we will need for k-th order estimator:
    # - the list of unique k+1 tuples of the meso-statesthat are possible
    # - the counts of these unique k+1 tuples over all paths
    # - the average time per lump over all paths

    # since we proceed in an online version, we simulate a batch, update the counts, and simulate again.

    # 1. we get the list of possilbe k+1 tuples of meso states that are possible.
    k_plus_1_unique_values = {} # this will be a mapping from order k to the list of k+1 tuples of meso states that are possible. That order of the list will be used for the count arrays.
    # we can't just call get_list_of_possible_transitions on the transition_prob_array, because that is for microstates
    # thus, we first create a "dummy" meso prob array, which is 1 at index i,j whenever there is a direct transition possible from meso state i to meso state j. 0 else.
    dummy_meso_prob_array = jnp.zeros((n_meso_states,n_meso_states))
    for i in range(n_micro_states):
        for j in range(n_micro_states):
            if transition_prob_array[i,j] != 0:
                dummy_meso_prob_array = dummy_meso_prob_array.at[
                    micro_to_meso_array[i],micro_to_meso_array[j]
                ].set(1)
    canonical_index_lookups = {}
    for estimator_order_k in estimator_order_abs_values:
        k_plus_1_unique_values[estimator_order_k] = jnp.asarray(
            get_list_of_possible_transitions(dummy_meso_prob_array,estimator_order_k+1),
            dtype=jnp.int32,
        )
        lookup_keys,lookup_indices = _create_canonical_index_lookup(
            k_plus_1_unique_values[estimator_order_k],n_meso_states
        )
        canonical_index_lookups[estimator_order_k] = (
            jax.device_put(lookup_keys,counting_device),
            jax.device_put(lookup_indices,counting_device),
        )
        print(f"For order k={estimator_order_k}, the number of unique k+1 tuples of meso states that are possible is {len(k_plus_1_unique_values[estimator_order_k])}.")

    micro_direct_transitions_unique_values = jnp.asarray(
        get_list_of_possible_transitions(transition_prob_array,2),dtype=jnp.int32
    ) # for the microscopic EPR
    micro_lookup_keys,micro_lookup_indices = _create_canonical_index_lookup(
        micro_direct_transitions_unique_values,n_micro_states
    )
    micro_lookup_keys = jax.device_put(micro_lookup_keys,counting_device)
    micro_lookup_indices = jax.device_put(micro_lookup_indices,counting_device)


    # The following maps order k to an array of canonical counts. Index i is
    # always the count of k_plus_1_unique_values[k][i].
    k_plus_1_counts_over_all_paths = {
        estimator_order_k:jax.device_put(
            jnp.zeros(
                len(k_plus_1_unique_values[estimator_order_k]),
                dtype=count_dtype,
            ),
            counting_device,
        )
        for estimator_order_k in estimator_order_abs_values
    }
    micro_direct_transitions_counts_over_all_paths = jax.device_put(
        jnp.zeros(
            len(micro_direct_transitions_unique_values),dtype=count_dtype
        ),
        counting_device,
    )

    # Only four scalars are retained for residence-time averaging. This gives
    # the weighted average over all lumps, even when batches contain different
    # numbers of mesoscopic lumps.
    total_micro_time = jax.device_put(
        jnp.zeros((),dtype=time_dtype),counting_device
    )
    number_of_micro_lumps = jax.device_put(
        jnp.zeros((),dtype=count_dtype),counting_device
    )
    total_meso_time = jax.device_put(
        jnp.zeros((),dtype=time_dtype),counting_device
    )
    number_of_meso_lumps = jax.device_put(
        jnp.zeros((),dtype=count_dtype),counting_device
    )

    micro_counting_time = 0.0
    times_counting_per_k = {
        estimator_order_k:0.0
        for estimator_order_k in estimator_order_abs_values
    }
    batch_keys = jr.split(key,N_batches)
    for batch_index in range(N_batches):
        print(f"[Info] Simulating batch {batch_index+1} of {N_batches}...")
        keys_per_simulation = jr.split(batch_keys[batch_index],N_paths_per_batch)
        simulation_start = time.perf_counter()
        micro_state_tracker_arrays,micro_time_tracker_arrays = simulator(keys_per_simulation)
        jax.block_until_ready((micro_state_tracker_arrays,micro_time_tracker_arrays))
        simulation_time = time.perf_counter() - simulation_start
        print(f"[Info] Simulation of batch {batch_index+1} took {simulation_time:.3f} seconds.")
        # we  check the shape
        assert micro_state_tracker_arrays.shape == (N_paths_per_batch,max_steps_per_path-steps_warmup,1), f"micro_state_tracker_arrays has shape {micro_state_tracker_arrays.shape}, but expected {(N_paths_per_batch,max_steps_per_path-steps_warmup,1)}"

        print(f"[Info]\tSumming dwell times and counting lumps...")
        micro_time_this_batch,micro_lumps_this_batch = (
            _sum_lump_times_and_count(
                micro_state_tracker_arrays,
                micro_time_tracker_arrays,
                count_dtype=count_dtype,
            )
        )
        total_micro_time += micro_time_this_batch
        number_of_micro_lumps += micro_lumps_this_batch

        print(f"[Info]\tCounting for microscopic EPR...")
        micro_counting_start = time.perf_counter()
        micro_direct_transitions_counts_this_batch = (
            _count_canonical_tuples_with_lookup(
                micro_state_tracker_arrays[:,:,0],
                micro_lookup_keys,
                micro_lookup_indices,
                micro_direct_transitions_unique_values.shape[0],
                n_micro_states,
                2,
                counting_chunk_size,
                count_dtype,
            )
        )
        micro_direct_transitions_counts_over_all_paths += micro_direct_transitions_counts_this_batch
        jax.block_until_ready(micro_direct_transitions_counts_over_all_paths)
        micro_counting_time += time.perf_counter()-micro_counting_start

        # now we coarse grain
        print(f"[Info]\tCoarse-graining")
        meso_state_tracker_arrays,meso_time_tracker_arrays = cg.micro_to_meso_batched(micro_state_tracker_arrays,micro_time_tracker_arrays,micro_to_meso_array)
        meso_time_this_batch,meso_lumps_this_batch = _sum_lump_times_and_count(
            meso_state_tracker_arrays,
            meso_time_tracker_arrays,
            count_dtype=count_dtype,
        )
        total_meso_time += meso_time_this_batch
        number_of_meso_lumps += meso_lumps_this_batch

        print(f"[Info]\tCounting for estimators of order k in {estimator_order_abs_values}...")
        for estimator_order_k in estimator_order_abs_values:
            counting_start = time.perf_counter()
            lookup_keys,lookup_indices = canonical_index_lookups[estimator_order_k]
            counts_this_batch = _count_canonical_tuples_with_lookup(
                meso_state_tracker_arrays,
                lookup_keys,
                lookup_indices,
                k_plus_1_unique_values[estimator_order_k].shape[0],
                n_meso_states,
                estimator_order_k+1,
                counting_chunk_size,
                count_dtype,
            )
            k_plus_1_counts_over_all_paths[estimator_order_k] += counts_this_batch
            jax.block_until_ready(k_plus_1_counts_over_all_paths[estimator_order_k])
            counting_time = time.perf_counter()-counting_start
            times_counting_per_k[estimator_order_k] += counting_time
            print(
                f"[Info] Counting order k={estimator_order_k} took "
                f"{counting_time:.3f} seconds."
            )

        del (
            micro_state_tracker_arrays,
            micro_time_tracker_arrays,
            meso_state_tracker_arrays,
            meso_time_tracker_arrays,
        )

    average_time_start = time.perf_counter()
    if int(jax.device_get(number_of_micro_lumps)) == 0:
        raise ValueError("No microscopic lumps were recorded")
    if int(jax.device_get(number_of_meso_lumps)) == 0:
        raise ValueError("No mesoscopic lumps were recorded")
    avg_micro_time_per_lump = total_micro_time/number_of_micro_lumps
    avg_meso_time_per_lump = total_meso_time/number_of_meso_lumps
    jax.block_until_ready((avg_micro_time_per_lump,avg_meso_time_per_lump))
    average_time_calculation_time = time.perf_counter()-average_time_start

    ret_estimators = {}

    # we first do the microscopic EPR
    microscopic_estimator_start = time.perf_counter()
    dS_micro = ests.dSk_est(
        micro_direct_transitions_counts_over_all_paths,
        micro_direct_transitions_unique_values,
        1,
        avg_micro_time_per_lump,
    )
    jax.block_until_ready(dS_micro)
    microscopic_estimator_time = (
        time.perf_counter()-microscopic_estimator_start
    )
    ret_estimators["dS_0"] = dS_micro
    print(f"[Info] Microscopic EPR estimation: dS_0 = {dS_micro}.")
    print(
        "[Info] Total time for dS_0 was "
        f"{(micro_counting_time+average_time_calculation_time+microscopic_estimator_time):.2f} "
        "seconds: "
        f"{micro_counting_time:.2f} counting, "
        f"{average_time_calculation_time:.2f} averaging dwell times, and "
        f"{microscopic_estimator_time:.2f} processing the counts."
    )


    # now we determine the k-th order EPR
    for signed_estimator_order_k in estimators_to_use:
        estimator_order_k = abs(signed_estimator_order_k)

        # The mean mesoscopic residence time is independent of estimator order.
        avg_time_per_lump_this_k = avg_meso_time_per_lump

        # now we call the estimator function
        conditioned_estimator_start = time.perf_counter()
        dSk = ests.dSk_est(
            k_plus_1_counts_over_all_paths[estimator_order_k],
            k_plus_1_unique_values[estimator_order_k],
            estimator_order_k,
            avg_time_per_lump_this_k,
        )
        jax.block_until_ready(dSk)
        conditioned_estimator_time = (
            time.perf_counter()-conditioned_estimator_start
        )

        # now we print the time it took to determine the k-th order estimator
        # the time is: time for counting the k+1 tuples + time for determining the average time per lump + time for calling the estimator function
        counting_time = times_counting_per_k[estimator_order_k]
        total_conditioned_time = (
            counting_time
            + average_time_calculation_time
            + conditioned_estimator_time
        )
        print(f"[Info] For order k={estimator_order_k}, dS_{estimator_order_k} = {dSk}.")
        print(
            f"[Info] Total time for dS_{estimator_order_k} was "
            f"{total_conditioned_time:.2f} seconds: "
            f"{counting_time:.2f} counting, "
            f"{average_time_calculation_time:.2f} averaging dwell times, and "
            f"{conditioned_estimator_time:.2f} processing the counts."
        )

        if signed_estimator_order_k < 0:
            # now we determine the unconditioned estimator
            unconditioned_estimator_start = time.perf_counter()
            dSkUncond = ests.unconditioned_estimator(
                k_plus_1_counts_over_all_paths[estimator_order_k],
                k_plus_1_unique_values[estimator_order_k],
                estimator_order_k,
                avg_time_per_lump_this_k,
            )
            jax.block_until_ready(dSkUncond)
            unconditioned_estimator_time = (
                time.perf_counter()-unconditioned_estimator_start
            )

            # now we print the time it took to determine the unconditioned estimator
            # the time is: time for counting the k+1 tuples + time for determining the average time per lump + time for calling the unconditioned estimator function
            total_unconditioned_time = (
                counting_time
                + average_time_calculation_time
                + unconditioned_estimator_time
            )
            print(
                f"[Info] For order k={estimator_order_k}, "
                f"dS_{estimator_order_k},uncond = {dSkUncond}."
            )
            print(
                f"[Info] Total time for dS_{estimator_order_k},uncond was "
                f"{total_unconditioned_time:.2f} seconds: "
                f"{counting_time:.2f} counting, "
                f"{average_time_calculation_time:.2f} averaging dwell times, and "
                f"{unconditioned_estimator_time:.2f} processing the counts."
            )

        # now we store the estimator(s) value
        ret_estimators[f"dS_{estimator_order_k}"] = dSk
        if signed_estimator_order_k < 0:
            ret_estimators[f"dS_{estimator_order_k},uncond"] = dSkUncond

    # now we return the estimators
    return ret_estimators


def _test_get_list_of_possible_transitions():
    transition_prob_array = jnp.array([
        [2.0,0.5,0.5,0.0],
        [0.5,2.0,0.0,0.5],
        [0.5,0.0,2.0,0.5],
        [0.0,0.5,0.5,2.0],
    ])

    expected_transitions_length_1 = [(0,),(1,),(2,),(3,)]
    possible_transitions_length_1 = get_list_of_possible_transitions(transition_prob_array,1)
    assert possible_transitions_length_1 == expected_transitions_length_1, f"Transitions of length 1 are not equal to the expected transitions. Expected: {expected_transitions_length_1}, Got: {possible_transitions_length_1}"

    expected_transitions_length_2 = [
        (0,1),(0,2),
        (1,0),(1,3),
        (2,0),(2,3),
        (3,1),(3,2),
    ]
    possible_transitions_length_2 = get_list_of_possible_transitions(transition_prob_array,2)
    assert possible_transitions_length_2 == expected_transitions_length_2, f"Transitions of length 2 are not equal to the expected transitions. Expected: {expected_transitions_length_2}, Got: {possible_transitions_length_2}"

    expected_transitions_length_3 = [
        (0,1,0),(0,1,3),(0,2,0),(0,2,3),
        (1,0,1),(1,0,2),(1,3,1),(1,3,2),
        (2,0,1),(2,0,2),(2,3,1),(2,3,2),
        (3,1,0),(3,1,3),(3,2,0),(3,2,3),
    ]
    possible_transitions_length_3 = get_list_of_possible_transitions(transition_prob_array,3)
    assert possible_transitions_length_3 == expected_transitions_length_3, f"Transitions of length 3 are not equal to the expected transitions. Expected: {expected_transitions_length_3}, Got: {possible_transitions_length_3}"
    print("[Test]  passed: get_list_of_possible_transitions")


def _test_count_dtype_selection():
    int32_max = onp.iinfo(onp.int32).max
    assert _get_count_dtype(int32_max) == jnp.int32
    if jax.config.x64_enabled:
        assert _get_count_dtype(int32_max+1) == jnp.int64
    else:
        try:
            _get_count_dtype(int32_max+1)
            assert False, "Counts above int32 should require JAX x64"
        except ValueError:
            pass
    print("[Test]  passed: count_dtype_selection")


def _test_count_canonical_tuples():
    transition_prob_array = jnp.array([
        [2.0,0.5,0.5,0.0],
        [0.5,2.0,0.0,0.5],
        [0.5,0.0,2.0,0.5],
        [0.0,0.5,0.5,2.0],
    ])
    canonical_tuples = get_list_of_possible_transitions(
        transition_prob_array,3
    )
    paths_states = jnp.array([
        [0,1,0,2,3,1,-1],
        [3,2,0,1,3,-1,-1],
        [1,0,-1,-1,-1,-1,-1],
    ],dtype=jnp.int32)

    tuple_to_index = {
        tuple_value:index for index,tuple_value in enumerate(canonical_tuples)
    }
    expected_counts = onp.zeros(len(canonical_tuples),dtype=onp.int32)
    for path in onp.asarray(paths_states):
        for start in range(path.shape[0]-2):
            tuple_value = tuple(path[start:start+3].tolist())
            if -1 not in tuple_value:
                expected_counts[tuple_to_index[tuple_value]] += 1

    counts = count_canonical_tuples(
        paths_states,
        canonical_tuples,
        4,
        chunk_size=2,
    )
    assert jnp.array_equal(counts,expected_counts), f"Counts differ. Expected {expected_counts}, got {counts}"

    if jax.config.x64_enabled:
        lookup_keys,lookup_indices = _create_canonical_index_lookup(
            canonical_tuples,4
        )
        int64_counts = _count_canonical_tuples_with_lookup(
            paths_states,
            lookup_keys,
            lookup_indices,
            len(canonical_tuples),
            4,
            3,
            2,
            jnp.int64,
        )
        assert int64_counts.dtype == jnp.int64
        assert jnp.array_equal(int64_counts,expected_counts)

    # Codes 1 and 5 have the same initial hash position in a four-slot lookup
    # table. This checks collision resolution and a deliberately reversed
    # canonical output order.
    colliding_canonical_tuples = [(0,5),(0,1)]
    colliding_paths = jnp.array([
        [0,1,-1],
        [0,5,-1],
        [0,5,-1],
    ],dtype=jnp.int32)
    colliding_counts = count_canonical_tuples(
        colliding_paths,colliding_canonical_tuples,100,chunk_size=1
    )
    assert jnp.array_equal(colliding_counts,jnp.array([2,1],dtype=jnp.int32)), f"Collision lookup counts differ. Expected [2,1], got {colliding_counts}"

    invalid_paths = jnp.array([[0,1,1]],dtype=jnp.int32)
    canonical_pairs = get_list_of_possible_transitions(
        transition_prob_array,2
    )
    try:
        count_canonical_tuples(
            invalid_paths,canonical_pairs,4,chunk_size=2
        )
        assert False, "A noncanonical observed tuple should raise a ValueError"
    except ValueError:
        pass
    print("[Test]  passed: count_canonical_tuples")


def _test_sum_lump_times_and_count():
    first_states = jnp.array([[0,1,-1]],dtype=jnp.int32)
    first_times = jnp.array([[1.0,3.0,0.0]],dtype=jnp.float32)
    second_states = jnp.array([
        [2,-1,-1],
        [1,0,2],
    ],dtype=jnp.int32)
    second_times = jnp.array([
        [10.0,0.0,0.0],
        [2.0,2.0,2.0],
    ],dtype=jnp.float32)

    first_total,first_count = _sum_lump_times_and_count(
        first_states,first_times
    )
    second_total,second_count = _sum_lump_times_and_count(
        second_states,second_times
    )
    average_time = (first_total+second_total)/(first_count+second_count)
    assert jnp.allclose(average_time,20/6), (
        f"Weighted average time differs. Expected {20/6}, got {average_time}"
    )
    if jax.config.x64_enabled:
        total_time_64,lump_count_64 = _sum_lump_times_and_count(
            second_states,second_times,count_dtype=jnp.int64
        )
        assert total_time_64.dtype == jnp.float64
        assert lump_count_64.dtype == jnp.int64
    print("[Test]  passed: sum_lump_times_and_count")


def _test_estimators_with_canonical_counts():
    # The final reverse pair is legal but unobserved. Canonical counting keeps
    # those zero bins, and both estimators must ignore them.
    canonical_tuples = jnp.array([
        [0,1],
        [1,0],
        [0,2],
        [2,0],
        [1,2],
        [2,1],
        [0,3],
        [3,0],
    ],dtype=jnp.int32)
    counts = jnp.array([30,10,15,5,20,10,0,0],dtype=jnp.int32)
    average_time = jnp.float32(2)

    conditioned = ests.dSk_est(
        counts,canonical_tuples,1,average_time
    )
    unconditioned = ests.unconditioned_estimator(
        counts,canonical_tuples,1,average_time
    )
    expected_conditioned = onp.log(2)/9
    expected_unconditioned = onp.log(3)/6 + onp.log(2)/18
    assert jnp.allclose(conditioned,expected_conditioned), (
        f"Conditioned estimate differs. Expected {expected_conditioned}, "
        f"got {conditioned}"
    )
    assert jnp.allclose(unconditioned,expected_unconditioned), (
        f"Unconditioned estimate differs. Expected {expected_unconditioned}, "
        f"got {unconditioned}"
    )

    try:
        ests.dSk_est(
            jnp.array([3,0],dtype=jnp.int32),
            jnp.array([[0,1],[1,0]],dtype=jnp.int32),
            1,
            average_time,
        )
        assert False, "An active tuple with a zero-count reverse should fail"
    except ValueError:
        pass

    # Order ten is part of the target workload. Each prefix below has a single
    # continuation, so its conditioned estimate is exactly zero.
    order_ten_tuple = jnp.array(
        [0,1,2,0,1,2,0,1,2,0,1],dtype=jnp.int32
    )
    order_ten_tuples = jnp.stack((order_ten_tuple,order_ten_tuple[::-1]))
    order_ten_counts = jnp.array([7,5],dtype=jnp.int32)
    order_ten_conditioned = ests.dSk_est(
        order_ten_counts,order_ten_tuples,10,average_time
    )
    order_ten_unconditioned = ests.unconditioned_estimator(
        order_ten_counts,order_ten_tuples,10,average_time
    )
    assert jnp.allclose(order_ten_conditioned,0), (
        f"Expected zero conditioned order-ten estimate, got "
        f"{order_ten_conditioned}"
    )
    assert jnp.allclose(order_ten_unconditioned,onp.log(7/5)/12), (
        f"Order-ten unconditioned estimate differs. Expected "
        f"{onp.log(7/5)/12}, got {order_ten_unconditioned}"
    )

    if jax.config.x64_enabled:
        large_counts = jnp.array(
            [2**32,2**32-1],dtype=jnp.int64
        )
        reverse_pair = jnp.array([[0,1],[1,0]],dtype=jnp.int32)
        large_conditioned = ests.dSk_est(
            large_counts,reverse_pair,1,jnp.float32(1)
        )
        large_unconditioned = ests.unconditioned_estimator(
            large_counts,reverse_pair,1,jnp.float32(1)
        )
        expected_large_unconditioned = (
            (large_counts[0]-large_counts[1])
            * jnp.log(large_counts[0]/large_counts[1])
            / jnp.sum(large_counts)
        )
        assert large_conditioned.dtype == jnp.float64
        assert large_unconditioned.dtype == jnp.float64
        assert jnp.allclose(large_conditioned,0)
        assert jnp.allclose(
            large_unconditioned,expected_large_unconditioned
        )
    print("[Test]  passed: estimators_with_canonical_counts")


def _test_all():
    _test_get_list_of_possible_transitions()
    _test_count_dtype_selection()
    _test_count_canonical_tuples()
    _test_sum_lump_times_and_count()
    _test_estimators_with_canonical_counts()
    print("[Test]  passed: all estimator_reworked.py tests")


if __name__ == "__main__":
    _test_all()
