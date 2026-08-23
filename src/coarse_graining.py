import jax
import jax.numpy as jnp
import jax.lax as lax


def scan_micro_to_meso_one_path(path_micro_states,path_micro_times,micro_to_meso_map):
    """
    Input:
        path_micro_states: array of shape (n_recorded_steps) which contains the history of the micro states.
        path_micro_times: array of shape (n_recorded_steps) which contains the history of the time spent in each micro state.
        micro_to_meso_map: array of shape (n_states,). It contains at index i the meso state number of the micro state i.
    Output:
        path_meso_states: array of shape (n_recorded_steps) which contains the history of the meso states.
        path_meso_times: array of shape (n_recorded_steps) which contains the history of the time spent in each meso state.
    """
    n_recorded_steps = path_micro_states.shape[0]
    path_meso_states = jnp.ones(shape=(n_recorded_steps), dtype=jnp.int32)*-1
    path_meso_times = jnp.zeros(shape=(n_recorded_steps), dtype=jnp.float32)
    if len(path_micro_states.shape) == 2:
        if path_micro_states.shape[1] != 1:
            raise ValueError(f"path_micro_states has shape {path_micro_states.shape} but should have shape (n_recorded_steps,1) aka ({n_recorded_steps},1)")
        else:
            path_micro_states = path_micro_states.flatten()
    if len(path_micro_times.shape) == 2:
        if path_micro_times.shape[1] != 1:
            raise ValueError(f"path_micro_times has shape {path_micro_times.shape} but should have shape (n_recorded_steps,1)")
        else:
            path_micro_times = path_micro_times.flatten()

    def _scan_body(carry,pair):
        (prev_meso_state,path_meso_states,path_meso_times,meso_saving_index) = carry
        current_micro_state, current_micro_time = pair
        current_meso_state = micro_to_meso_map[current_micro_state]

        # if our system is a new mesostate, we write to next index
        meso_saving_index = jnp.where(prev_meso_state != current_meso_state,meso_saving_index + 1,meso_saving_index)
        
        # update the records at index meso_saving_index
        path_meso_states = path_meso_states.at[meso_saving_index].set(current_meso_state)
        path_meso_times = path_meso_times.at[meso_saving_index].add(current_micro_time)

        return (current_meso_state,path_meso_states,path_meso_times,meso_saving_index),None
    

    carry_init = (-1, path_meso_states, path_meso_times,-1)
    final_carry,_ = lax.scan(_scan_body, carry_init, (path_micro_states, path_micro_times))
    _,final_meso_states,final_meso_times,_ = final_carry
    return final_meso_states, final_meso_times


def micro_to_meso_batched(paths_micro_states,paths_micro_times,micro_to_meso_map):
    """
    Input:
        paths_micro_states: array of shape (n_paths,n_recorded_steps) which contains the history of the micro states.
        paths_micro_times: array of shape (n_paths,n_recorded_steps) which contains the history of the time spent in each micro state.
        micro_to_meso_map: array of shape (n_states,). It contains at index i the meso state number of the micro state i.
    Output:
        paths_meso_states: array of shape (n_paths,n_meso_steps) which contains the history of the meso states.
        paths_meso_times: array of shape (n_paths,n_meso_steps) which contains the history of the time spent in each meso state.
    Note:
        n_meso_steps is determined.
    """
    # we just vmap scan_micro_to_meso_one_path. The micro_to_meso_map is the same for all paths.
    paths_meso_states, paths_meso_times = jax.vmap(scan_micro_to_meso_one_path, in_axes=(0,0,None))(paths_micro_states,paths_micro_times,micro_to_meso_map)
    
    # reshape
    # check in paths_meso_states for each column whether it contains only -1
    all_minus_one = jnp.all(paths_meso_states == -1,axis=0)
    # get the index of the first false entry from the right
    last_index_where_all_are_not_minus_one = jnp.where(all_minus_one == False)[0][-1]
    # reshape paths and times of meso states to cut the minus one entries
    paths_meso_states = paths_meso_states[:,:last_index_where_all_are_not_minus_one+1]
    paths_meso_times = paths_meso_times[:,:last_index_where_all_are_not_minus_one+1]
    #print(f"Paths meso states shape: {paths_meso_states.shape}, paths meso times shape: {paths_meso_times.shape}")
    return paths_meso_states, paths_meso_times

""" def count_occurances(paths_states):
    Input:
        paths_states: array of shape (n_paths,n_recorded_steps) which contains the history of the states.
    Output:
        counts: array of dictionary with keys the states and values the counts.
    # we use jnp.unique_counts on each path
    def _one_path_count_occurances(path_states):
        unique_values,unique_counts = jnp.unique_counts(path_states)
        return dict(zip(unique_values,unique_counts))
    counts = jax.vmap(_one_path_count_occurances)(paths_states)
    return counts """

def _test_scan_micro_to_meso_one_path():
    micro_state_tracker_arrays = jnp.array([0,1,2,3,4,3])
    micro_time_tracker_arrays = jnp.array([2.0,3.0,4.0,5.0,6.0,7.0])
    micro_to_meso_array = jnp.array([0,0,1,1,2])
    expected_meso_states = jnp.array([0,1,2,1,-1,-1])
    expected_meso_times = jnp.array([5.0,9.0,6.0,7.0,0,0.0])
    is_meso_states,is_meso_times = scan_micro_to_meso_one_path(micro_state_tracker_arrays,micro_time_tracker_arrays,micro_to_meso_array)
    # check that the meso states and times are equal to the expected ones
    assert jnp.all(is_meso_states == expected_meso_states), f"Meso states are not equal to the expected ones. Expected: {expected_meso_states}, Got: {is_meso_states}"
    assert jnp.all(is_meso_times == expected_meso_times), f"Meso times are not equal to the expected ones. Expected: {expected_meso_times}, Got: {is_meso_times}"
    print("[Test]  passed: scan_micro_to_meso_one_path")

def _test_micro_to_meso_batched():
    micro_state_tracker_arrays = jnp.array([[0,1,2,3,4,4],[0,1,3,3,4,3],[0,1,2,3,4,1]])
    micro_time_tracker_arrays = jnp.array([[2.0,3.0,4.0,5.0,6.0,7.0],[2.0,3.0,4.0,5.0,6.0,7.0],[2.0,3.0,4.0,5.0,6.0,7.0]])
    micro_to_meso_array = jnp.array([0,0,1,1,2])
    expected_meso_states = jnp.array([[0,1,2,-1],[0,1,2,1],[0,1,2,0]])
    expected_meso_times = jnp.array([[5.0,9.0,13.0,0.0],[5.0,9.0,6.0,7.0],[5.0,9.0,6.0,7.0]])
    is_meso_states,is_meso_times = micro_to_meso_batched(micro_state_tracker_arrays,micro_time_tracker_arrays,micro_to_meso_array)
    # check that the meso states and times are equal to the expected ones
    assert jnp.all(is_meso_states == expected_meso_states), f"Meso states are not equal to the expected ones. Expected: {expected_meso_states}, Got: {is_meso_states}"
    assert jnp.all(is_meso_times == expected_meso_times), f"Meso times are not equal to the expected ones. Expected: {expected_meso_times}, Got: {is_meso_times}"
    print("[Test]  passed: micro_to_meso_batched")


def _test_all():
    _test_scan_micro_to_meso_one_path()
    _test_micro_to_meso_batched()
    print("[Test]  passed: all coarse_graining.py tests")


if __name__ == "__main__":
    _test_all()