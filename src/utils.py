import jax
import jax.numpy as jnp
import numpy as onp

def transition_list_to_prob_array(transition_list):
    """
    Input:
        transition_list: list of tuples (i,j,k) where i,j are the states and k is the rate of transitioning from i to j.
    Output:
        transition_matrix: array of shape (n_states,n_states). The diagonal contains the out_rates (positive!) off diagonals at location [i,j] the probability of transitioning from i to j.
    """
    n_states = max(max(i,j) for (i,j,k) in transition_list)+1
    transition_matrix = jnp.zeros((n_states,n_states))
    for (i,j,k) in transition_list:
        transition_matrix = transition_matrix.at[i,j].set(k)
    # get the sum of each row
    out_rates = jnp.sum(transition_matrix,axis=1)
    #print(f"Out rates: {out_rates}")

    # now, we want to divide each row by its out_rate
    transition_matrix = transition_matrix / out_rates[:,jnp.newaxis]
    
    # the diagonal is the out_rate
    for i in range(n_states):
        transition_matrix = transition_matrix.at[i,i].set(out_rates[i])
    return transition_matrix

def _test_transition_list_to_prob_array():
    test_transition_list = [
        (0,1,10),
        (0,2,20),
        (1,0,100),
        (1,2,100),
        (2,0,1000)
    ]

    prob_array_should = jnp.array([
        [30,1/3.0,2/3.0],
        [1/2.0,200,1/2.0],
        [1,0,1000]
    ])
    prob_array_is = transition_list_to_prob_array(test_transition_list)
    assert jnp.allclose(prob_array_is,prob_array_should), "Probability arrays are not equal"
    print("[Test]  passed: transition_list_to_prob_array")

def lump_list_to_micro_to_meso_map(lump_list):
    """
    Input:
        lump_list: list of lists, where each sublist contains the micro states that belong to the same meso state.
    Output:
        micro_to_meso_map: array of shape (n_states,). It contains at index i the meso state number of the micro state i.
    """
    max_state = max([max(m) for m in lump_list])
    n_states = max_state + 1
    micro_to_meso_map = -1*jnp.ones(n_states,dtype=jnp.int32)
    for i, lump in enumerate(lump_list):
        for micro_state in lump:
            # check that the micro state is not already set
            #print(f"Current micro to meso map: {micro_to_meso_map}")
            assert micro_to_meso_map[micro_state] == -1, f"Micro state {micro_state} is already set to meso state {micro_to_meso_map[micro_state]}."
            micro_to_meso_map = micro_to_meso_map.at[micro_state].set(i)
    return micro_to_meso_map

def paths_to_k_th_order(paths_states,k):
    """
    Input:
        paths_states: array of shape (n_paths,n_recorded_steps) which contains the history of the states.
    Output:
        k_th_order_paths: array of shape (n_paths,n_recorded_steps-k+1,k) which contains the history in form of (overlapping) subsequent k tuples along each path    
    Notes:
        None of the tuples will be partially incomplete (ie contain only some -1 values). If a tuple contains at least one -1 value, it is replaced by a tuple of -1 values.
    """
    # we basically stack for each path in the following form:
    # if path_0 = paths_states[0,:]
    # then for k = 2 k_th_order_paths[0] = jnp.stack([path_0[:-1],path_0[1:]], axis = 1)
    # for k=3, we would have k_th_order_paths[0] = jnp.stack([path_0[:-2],path_0[1:-1],path_0[2:]], axis = 1)
    # and so on.
    # we can do this in a vectorized way for all paths at once
    assert len(paths_states.shape) == 2, "paths_states must be a 2D array: (n_paths,n_recorded_steps)"
    n_paths = paths_states.shape[0]
    n_recorded_steps = paths_states.shape[1]

    # we first define the transform for one path, and then use jax.vmap to apply it to all paths
    def _one_path_transform(path):
        path_length = path.shape[0]
        return jnp.stack([path[i:(path_length-k+i+1)] for i in range(k)], axis=1)
    
    
    k_th_order_paths = jax.vmap(_one_path_transform)(paths_states)

    # each path has shape (n_recorded_steps-k+1,k). We now want to check which of the tuples contains at least one -1 value. For each such tuple, we replace all values in the tuple by -1
    def _one_path_partial_minus_one_to_full_minus_one(path):
        return jnp.where(jnp.any(path == -1,axis=1)[:,jnp.newaxis],-1*jnp.ones(k),path)
    k_th_order_paths = jax.vmap(_one_path_partial_minus_one_to_full_minus_one)(k_th_order_paths)
    return k_th_order_paths

def _test_paths_to_k_th_order_full_paths():
    paths_states = jnp.array([[0,1,2,3,4,3],[0,1,3,5,4,3],[0,1,2,3,4,1]])
    expected_2_order_paths = jnp.array(
        [[
            [0,1],
            [1,2],
            [2,3],
            [3,4],
            [4,3]
        ],
        [
            [0,1],
            [1,3],
            [3,5],
            [5,4],
            [4,3]
        ],
        [
            [0,1],
            [1,2],
            [2,3],
            [3,4],
            [4,1]
        ]
        ])
    k_th_order_paths = paths_to_k_th_order(paths_states,2)
    assert jnp.allclose(k_th_order_paths,expected_2_order_paths), "k_th_order_paths does not match expected_2_order_paths"
    expected_3_order_paths = jnp.array(
        [[
            [0,1,2],
            [1,2,3],
            [2,3,4],
            [3,4,3]
        ],
        [
            [0,1,3],
            [1,3,5],
            [3,5,4],
            [5,4,3]
        ],
        [
            [0,1,2],
            [1,2,3],
            [2,3,4],
            [3,4,1]
        ]
        ])
    k_th_order_paths = paths_to_k_th_order(paths_states,3)
    assert jnp.allclose(k_th_order_paths,expected_3_order_paths), "k_th_order_paths does not match expected_3_order_paths"
    print("[Test]  passed: paths_to_k_th_order (full paths)")



def _test_paths_to_k_th_order_partial_minus_one():
    # Paths containing -1 should turn any partially -1 tuple into an all -1 tuple
    paths_states = jnp.array([
        [0,1,-1,3,4],
        [0,-1,2,3,-1],
        [-1,1,2,3,4]
    ])

    # k = 2
    expected_2_order_paths = jnp.array([
        [
            [0,1],
            [-1,-1],
            [-1,-1],
            [3,4]
        ],
        [
            [-1,-1],
            [-1,-1],
            [2,3],
            [-1,-1]
        ],
        [
            [-1,-1],
            [1,2],
            [2,3],
            [3,4]
        ]
    ])
    k_th_order_paths = paths_to_k_th_order(paths_states,2)
    assert jnp.allclose(k_th_order_paths, expected_2_order_paths), "k=2: partial -1 tuples were not fully replaced with -1"

    # k = 3
    expected_3_order_paths = jnp.array([
        [
            [-1,-1,-1],
            [-1,-1,-1],
            [-1,-1,-1]
        ],
        [
            [-1,-1,-1],
            [-1,-1,-1],
            [-1,-1,-1]
        ],
        [
            [-1,-1,-1],
            [1,2,3],
            [2,3,4]
        ]
    ])
    k_th_order_paths = paths_to_k_th_order(paths_states,3)
    assert jnp.allclose(k_th_order_paths, expected_3_order_paths), "k=3: partial -1 tuples were not fully replaced with -1"
    print("[Test]  passed: paths_to_k_th_order (-1 replacement in partial paths)")


@jax.jit
def _count_occurances_given_unique_values_jitted(paths_states,unique_values,weights):
    minimum_value = jnp.minimum(jnp.min(paths_states),jnp.min(unique_values))
    shifted_paths_states = paths_states - minimum_value
    shifted_unique_values = unique_values - minimum_value
    path_codes = jnp.sum(shifted_paths_states*weights,axis=-1).reshape(-1)
    unique_codes = jnp.sum(shifted_unique_values*weights,axis=-1)

    unique_code_order = jnp.argsort(unique_codes)
    sorted_unique_codes = unique_codes[unique_code_order]
    sorted_positions = jnp.searchsorted(sorted_unique_codes,path_codes)
    safe_sorted_positions = jnp.clip(sorted_positions,0,unique_values.shape[0]-1)
    every_tuple_has_one_match = jnp.all(
        (sorted_positions < unique_values.shape[0])
        & (sorted_unique_codes[safe_sorted_positions] == path_codes)
    )
    unique_values_are_unique = jnp.all(sorted_unique_codes[1:] != sorted_unique_codes[:-1])
    original_positions = unique_code_order[safe_sorted_positions]
    counts = jnp.bincount(
        original_positions,
        length=unique_values.shape[0],
    )
    return counts,every_tuple_has_one_match,unique_values_are_unique


def count_occurances_given_unique_values(paths_states,unique_values):
    """
    Input:
        paths_states: array of shape (n_paths,n_recorded_steps,k) which contains tuples of states.
        unique_values: array-like of shape (n_unique_tuples,k) which contains every tuple that may occur. The order of this array determines the order of the returned counts.
    Output:
        counts: jnp array of shape (n_unique_tuples,). counts[i] is the total number of occurances of unique_values[i] across all paths.
    """
    paths_states = jnp.asarray(paths_states)
    unique_values = jnp.asarray(unique_values)
    assert len(paths_states.shape) == 3, "paths_states must be a 3D array: (n_paths,n_recorded_steps,k)"
    assert len(unique_values.shape) == 2, "unique_values must be a 2D array: (n_unique_tuples,k)"
    assert paths_states.shape[2] == unique_values.shape[1], f"The tuples in paths_states and unique_values must have the same length. Got {paths_states.shape[2]} and {unique_values.shape[1]}"
    if unique_values.shape[0] == 0:
        raise ValueError("unique_values must contain at least one tuple")

    min_value = int(jnp.minimum(jnp.min(paths_states),jnp.min(unique_values)))
    max_value = int(jnp.maximum(jnp.max(paths_states),jnp.max(unique_values)))
    base = max_value - min_value + 1
    tuple_length = paths_states.shape[2]
    max_code = base**tuple_length - 1
    code_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
    dtype_max = onp.iinfo(onp.int64 if jax.config.x64_enabled else onp.int32).max
    if max_code > dtype_max:
        raise ValueError(f"Tuple codes require values up to {max_code}, which do not fit in the enabled integer dtype")
    weights = base**jnp.arange(tuple_length-1,-1,-1,dtype=code_dtype)

    counts,every_tuple_has_one_match,unique_values_are_unique = _count_occurances_given_unique_values_jitted(
        paths_states.astype(code_dtype),unique_values.astype(code_dtype),weights
    )
    if not unique_values_are_unique:
        raise ValueError("unique_values must not contain duplicate tuples")
    if not every_tuple_has_one_match:
        raise ValueError("Every tuple in paths_states must occur exactly once in unique_values")
    return counts


def _test_count_occurances_given_unique_values():
    paths_states = jnp.array([
        [[0,1],[1,2],[0,1],[2,0]],
        [[1,2],[2,0],[2,0],[0,1]],
    ])
    unique_values = [(2,0),(0,1),(3,4),(1,2)]
    expected_counts = jnp.array([3,3,0,2],dtype=jnp.int32)
    counts = count_occurances_given_unique_values(paths_states,unique_values)
    assert jnp.array_equal(counts,expected_counts), f"Counts are not equal to the expected counts. Expected: {expected_counts}, Got: {counts}"

    paths_states_with_unknown_tuple = paths_states.at[0,0].set(jnp.array([4,5]))
    try:
        count_occurances_given_unique_values(paths_states_with_unknown_tuple,unique_values)
        assert False, "An unknown tuple should raise a ValueError"
    except ValueError:
        pass
    print("[Test]  passed: count_occurances_given_unique_values")

def count_occurances(paths_states):
    """
    Input:
        paths_states: array of shape (n_paths,n_recorded_steps,k) which contains the history of the states. Need n_paths >= 3
        The first path will be taken to determine which unique tuples occur.
        The second and third path will be taking to check that no tuples were missed out (in other words: that n_recorded_steps was sufficiently long for the given stochastic process)
        The statistics will be over all paths (including the first three ones)
    Output:
        counts: array of shape n_paths x n_unique_tuples. counts[n,i] contains the number of times the i-th tuple occurred in the n-th path.
        tuples: list of tuples, where the i-th tuple is the tuple that is counted in counts[:,i]
        sucess: true if the counting was successful, false otherwise (e.g. if the paths were too short so that some tuples of second_path_unique_values, third_path_unique_values, or last_path_unique_values are not in the first path unique values)
    """
    n_paths = paths_states.shape[0]
    k = paths_states.shape[2]
    assert n_paths >= 3, "n_paths must be at least 3"

    ## we take the first path's values as the unique values
    unique_values = jnp.unique(paths_states[0,:],axis=0)
    # the above is usually already sorted, but we do not want to rely on that (implicit) sorting being stable across future releases. hence:
    # now, we sort the unique values. They are of shape (n_unique_tuples,k). We sort them so that the lexicographic first tuple is first, then the second, etc.
    unique_values = unique_values[jnp.lexsort(unique_values.T[::-1])]
    #print(f"Determinig unique values from path of shape {(paths_states[0,:]).shape}\nwith first 10 elements: {paths_states[0,:10]}")
    #print(f"Unique values: {unique_values}")
    ## we then check that the second and third path's values are in the first path's values
    #print(f"First path unique values: {unique_values}")
    second_path_unique_values = jnp.unique(paths_states[1,:],axis=0)
    #print(f"Second path unique values: {second_path_unique_values}")
    third_path_unique_values = jnp.unique(paths_states[2,:],axis=0)
    #print(f"Third path unique values: {third_path_unique_values}")
    last_path_unique_values = jnp.unique(paths_states[-1,:],axis=0)
    #print(f"Last path unique values: {last_path_unique_values}")
    # check that each of the second and third path unique values are in the first path unique values
    def _tuple_in_list(given_tuple, given_list):
        # Convert to JAX arrays and ensure proper shapes
        given_tuple = jnp.array(given_tuple)
        given_list = jnp.array(given_list)
        
        # Check if tuple is in list by comparing each row
        return jnp.any(jnp.all(given_tuple == given_list, axis=1))

    # checks if all tuples in list_of_tuples are in given_list
    def _all_tuples_in_list(list_of_tuples,given_list):
        return jnp.all(jax.vmap(_tuple_in_list,in_axes=(0,None))(list_of_tuples,given_list))

    # in detail, we check the last path.
    if not  _all_tuples_in_list(last_path_unique_values,unique_values):
        # find the first tuple in last_path_unique_values that is not in unique_values
        is_in_unique_values = jax.vmap(_tuple_in_list,in_axes=(0,None))(last_path_unique_values,unique_values)
        first_not_in_unique_values = jnp.where(is_in_unique_values == False)[0][0]
        first_not_in_unique_values_tuple = last_path_unique_values[first_not_in_unique_values]
        print(f"First not in unique values tuple: {first_not_in_unique_values_tuple}")
        print(f"Unique values: {unique_values}")
        print(f"[Warning] when counting tuples of length {k}, the path at index -1 (last path) unique values are not in the first path unique values. That means that the paths are too short (n_recorded_steps must be increased)")
        return None,None,False

    # checking second and third path as well
    if not  _all_tuples_in_list(second_path_unique_values,unique_values):
        # find the first tuple in last_path_unique_values that is not in unique_values
        is_in_unique_values = jax.vmap(_tuple_in_list,in_axes=(0,None))(second_path_unique_values,unique_values)
        first_not_in_unique_values = jnp.where(is_in_unique_values == False)[0][0]
        first_not_in_unique_values_tuple = second_path_unique_values[first_not_in_unique_values]
        print(f"First not in unique values tuple: {first_not_in_unique_values_tuple}")
        print(f"Unique values: {unique_values}")
        print(f"[Warning] when counting tuples of length {k}, the path at index 1 (second path) unique values are not in the first path unique values. That means that the paths are too short (n_recorded_steps must be increased)")
        return None,None,False
    if not _all_tuples_in_list(third_path_unique_values,unique_values):
        print(f"[Warning] when counting tuples of length {k}, the path at index 2 (third path) unique values are not in the first path unique values. That means that the paths are too short (n_recorded_steps must be increased)")
        return None,None,False

    def _count_on_k_th_order_path(this_path_states,unique_values):
        #print(f"this_path_states: {this_path_states}")
        # this_path_states has shape (n_recorded_steps-k+1,k)
        this_path_unique_values,this_path_unique_counts = jnp.unique(this_path_states,axis=0,size = unique_values.shape[0],return_counts=True,fill_value=-1*jnp.ones(k))
        #print(f"This path unique counts: {this_path_unique_counts}")
        # now, we bring the counts into the correct order (the order of unique_values)
        out_count = jnp.zeros(unique_values.shape[0],dtype=jnp.int32)
        # we now get for each tuple in this_path_unique_values the index in unique_values
        def _tuple_to_first_index(tuple):
            return jnp.where(jnp.all(tuple == unique_values,axis=1),size=1)[0][0]
        indices = jax.vmap(_tuple_to_first_index,in_axes=(0))(this_path_unique_values)
        # now we index with that: out_count[indices[i]] = this_path_unique_counts[i]
        out_count = out_count.at[indices].set(this_path_unique_counts)
        return out_count

    counts = jax.vmap(_count_on_k_th_order_path,in_axes=(0,None))(paths_states,unique_values)

    return counts,unique_values, True

def _test_count_occurances():
    second_order_paths =jnp.array([
       [[ 0.,  1.],
        [ 1.,  2.],
        [ 2.,  3.],
        [ 3.,  4.],
        [ 4.,  3.],
        [ 3.,  4.],
        [ 4.,  3.],
        [ 3.,  3.],
        [ 3.,  1.],
        [ 1.,  3.],
        [ 3.,  2.],
        [ 2.,  1.],
        [ 1.,  4.],
        [ 4.,  1.],
        [-1., -1.]],
       [[ 0.,  1.],
        [ 1.,  3.],
        [ 3.,  3.],
        [ 3.,  4.],
        [ 4.,  3.],
        [ 3.,  4.],
        [ 4.,  3.],
        [ 3.,  4.],
        [ 4.,  3.],
        [ 3.,  3.],
        [ 3.,  3.],
        [ 3.,  3.],
        [ 3.,  3.],
        [ 3.,  3.],
        [ 3.,  3.]],
       [[ 0.,  1.],
        [ 1.,  2.],
        [ 2.,  3.],
        [ 3.,  4.],
        [ 4.,  1.],
        [ 1.,  2.],
        [ 2.,  1.],
        [ 1.,  2.],
        [ 2.,  1.],
        [ 1.,  2.],
        [ 2.,  1.],
        [ 1.,  2.],
        [ 2.,  1.],
        [ 1.,  2.],
        [ 2.,  1.]]])
    counts,unique_values,count_success = count_occurances(second_order_paths)
    assert count_success, "count_occurances should return True"
    tuple_to_index_dict = {tuple(onp.array(unique_values[i])):i for i in range(unique_values.shape[0])}
    assert jnp.allclose(counts[:,tuple_to_index_dict[(0,1)]], jnp.array([1,1,1])), "counts for tuple (0,1) are not correct"
    assert jnp.allclose(counts[:,tuple_to_index_dict[(1,2)]], jnp.array([1,0,6])), "counts for tuple (1,2) are not correct"
    assert jnp.allclose(counts[:,tuple_to_index_dict[(2,3)]], jnp.array([1,0,1])), "counts for tuple (2,3) are not correct"
    assert jnp.allclose(counts[:,tuple_to_index_dict[(3,4)]], jnp.array([2,3,1])), "counts for tuple (3,4) are not correct"
    assert jnp.allclose(counts[:,tuple_to_index_dict[(4,3)]], jnp.array([2,3,0])), "counts for tuple (4,3) are not correct"
    assert jnp.allclose(counts[:,tuple_to_index_dict[(3,3)]], jnp.array([1,7,0])), "counts for tuple (3,3) are not correct"
    assert jnp.allclose(counts[:,tuple_to_index_dict[(-1,-1)]], jnp.array([1,0,0])), "counts for tuple (-1,-1) are not correct"
    print("[Test]  passed: count_occurances")


def _test_all():
    _test_transition_list_to_prob_array()
    _test_paths_to_k_th_order_full_paths()
    _test_paths_to_k_th_order_partial_minus_one()
    _test_count_occurances_given_unique_values()
    _test_count_occurances()
    print("[Test]  passed: all utils.py tests")

if __name__ == "__main__":
    _test_all()
