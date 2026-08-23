


import jax.numpy as jnp
import jax
import jax.lax as lax
import jax.random as jr

from . import utils
from . import coarse_graining as cg

# we first simulate the system with given transition rates. 
# states are 0-indexed
def _single_step(carry,x):
    """
    carry: (transition_prob_array,steps_warmup,current_state, state_and_time_tracker_array,next_saving_index,step_count)
            The arguments have the following meaning:
            transition_prob_only_array,: array of shape (n_states,n_states). On off-diagonals, it contains at transition_prob_array[i,j] the probability of transitioning from state i to state j. Diagonal transition_prob_array[i,i] are all 0.
            diagonal_rates: array of shape (n_states,). It contains the rates of leaving each state.
            steps_warmup: int, the number of steps to warmup the system. During warmup, we do not record the states.
            current_state: int, the current state of the system. States are 0-indexed.
            micro_state_tracker_array: array of ints of shape (max_steps-steps_warmup,1). It contains the history of the micro states.
            micro_time_tracker_array: array of floats of shape (max_steps-steps_warmup,1). It contains the history of the time spent in each micro state.
            micro_next_saving_index: int, the next index in state_and_time_tracker_array to save the state and time in the state_and_time_tracker_array.
    x: key which will be split into (key_for_next_state,key_for_time_spent)
    """
    (transition_prob_only_array,diagonal_rates,steps_warmup,current_state, micro_state_tracker_array,micro_time_tracker_array,micro_next_saving_index,step_count) = carry

    (key_for_next_state,key_for_time_spent) = jr.split(x)
    # first, we determine the next state
    time_spent_in_current_state = jr.exponential(key_for_time_spent,dtype=jnp.float32)
    time_spent_in_current_state = (time_spent_in_current_state/diagonal_rates[current_state]).astype(jnp.float32)

    # update the micro records. If we are not recording, this just overwrites the 0th entry
    micro_state_tracker_array = micro_state_tracker_array.at[micro_next_saving_index,0].set(current_state)
    micro_time_tracker_array = micro_time_tracker_array.at[micro_next_saving_index,0].set(time_spent_in_current_state)
    micro_next_saving_index = jnp.where(step_count >= steps_warmup, micro_next_saving_index + 1, 0)


    #jax.debug.print("transition_prob_only_array[current_state,:]: {transition_rates}",transition_rates=transition_prob_only_array[current_state,:])
    next_state = jr.choice(key_for_next_state,a = jnp.arange(transition_prob_only_array.shape[0]),p = transition_prob_only_array[current_state,:])
    #jax.debug.print("next_state: {next_state}",next_state=next_state)
    # update current state
    current_state = next_state

    # update step count
    step_count += 1
    return (transition_prob_only_array,diagonal_rates,steps_warmup,current_state,micro_state_tracker_array,micro_time_tracker_array,micro_next_saving_index,step_count),None

def scan_one_path(
    init_key: jr.PRNGKey,
    transition_prob_array: jnp.ndarray,
    steps_warmup: int,
    max_steps: int,
):
    """
    init_key: jr.PRNGKey,
    transition_prob_array: jnp.ndarray, on off diagaonal at index [i,j] it contains the probability of transitioning from state i to state j. Diagonal at index [i,i] contains the exit rates of the state i.
    steps_warmup: int,
    max_steps: int: maximum number of steps to simulate, including warmup.

    Returns:
        state_and_time_tracker_array: array of shape (max_steps-steps_warmup,2). It contains the history of the states and the time spent in each state.
    """
    init_state = 0 # with warmup, we may start in any state
    micro_state_tracker_array = jnp.zeros((max_steps-steps_warmup,1),dtype=jnp.int32)
    micro_time_tracker_array = jnp.zeros((max_steps-steps_warmup,1),dtype=jnp.float32)
    
    # diagonal_rates is the diagonal of transition_prob_array
    diagonal_rates = jnp.diag(transition_prob_array)
    # transition_prob_only_array is like transition_prob_array, but without the diagonal rates
    transition_prob_only_array = transition_prob_array - jnp.diag(diagonal_rates)

    """ jax.debug.print("transition_prob_array: {transition_prob_array}",transition_prob_array=transition_prob_array)
    jax.debug.print("transition_prob_only_array: {transition_prob_only_array}",transition_prob_only_array=transition_prob_only_array)
    jax.debug.print("diagonal_rates: {diagonal_rates}",diagonal_rates=diagonal_rates) """
    

    carry_init = (transition_prob_only_array,diagonal_rates,steps_warmup,init_state,micro_state_tracker_array,micro_time_tracker_array,0,0)

    keys = jr.split(init_key,max_steps)

    def _scan_body(carry,x):
        return _single_step(carry,x)
    
    final_carry,_ = lax.scan(_scan_body,carry_init,keys,length=max_steps)

    micro_state_tracker_array = final_carry[4]
    micro_time_tracker_array = final_carry[5]
    return micro_state_tracker_array,micro_time_tracker_array


def get_simulator_batched(transition_prob_array,steps_warmup,max_steps):
    """
    transition_prob_array: array of shape (n_states,n_states). On off-diagonals, it contains at transition_prob_array[i,j] the probability of transitioning from state i to state j. Diagonal transition_prob_array[i,i] contains the exit rates of the state i.
    steps_warmup: int, the number of steps to warmup the system. During warmup, we do not record the states.
    max_steps: int: maximum number of steps to simulate, including warmup.
    Returns:
        micro_state_tracker_arrays: array of shape (N_parallel_paths,max_steps-steps_warmup,1). It contains the history of the micro states.
        micro_time_tracker_arrays: array of shape (N_parallel_paths,max_steps-steps_warmup,1). It contains the history of the time spent in each micro state.
    """
    @jax.jit
    def _single_path_wrapper(key):
        return scan_one_path(key,transition_prob_array,steps_warmup,max_steps)
    
    _batched_simulator = jax.vmap(_single_path_wrapper,in_axes=(0))
    batched_simulator_jitted = jax.jit(_batched_simulator)
    
    return batched_simulator_jitted



def _test_simulator_batched():
    w = 2
    u = 2
    k = 1
    v = 1

    steps_warmup = 1*10**3
    max_steps = 10**4
    N_parallel_paths = 10**2

    n_micro_states  = 6
    transition_list = []
    for i in range(0, n_micro_states):
        # next state is i+1 but modulo n_micro_states
        next_state = (i+1) % n_micro_states
        previous_state = (i-1) % n_micro_states
        if i % 2 == 1: # 1, 3, ...
            transition_list.append((i, next_state, w))
            transition_list.append((i, previous_state, v))
        else: # 2, 4, ...
            transition_list.append((i, next_state, u))
            transition_list.append((i, previous_state, k))
    lumps = [[i,i+1] for i in range(0, n_micro_states,2)]
    #print(f"Lumps: {lumps}")
    micro_to_meso_array = utils.lump_list_to_micro_to_meso_map(lumps)
    #print(f"Micro to meso map:\n{micro_to_meso_array}")
    transition_prob_array = utils.transition_list_to_prob_array(transition_list)
    #print(f"Transition probability array:\n{transition_prob_array}")
    simulator = get_simulator_batched(transition_prob_array,steps_warmup,max_steps)
    # run the simulator
    keys = jr.split(jr.PRNGKey(0),N_parallel_paths)
    micro_state_tracker_arrays,micro_time_tracker_arrays = simulator(keys)
    #print(f"State and time tracker arrays shape: {micro_state_tracker_arrays.shape}, {micro_time_tracker_arrays.shape}")
    trivial_lumping_paths_meso_states, trivial_lumping_paths_meso_times = cg.micro_to_meso_batched(micro_state_tracker_arrays,micro_time_tracker_arrays,jnp.array([0,1,2,3,4,5]))
    # the trivial lumping states and time should be the same as the micro states and times
    assert jnp.allclose(trivial_lumping_paths_meso_states,micro_state_tracker_arrays[:,:,0]), "[test]  get_simulator_batched is not working as expected: states of trivial lumping differ from micro states"
    assert jnp.allclose(trivial_lumping_paths_meso_times,micro_time_tracker_arrays[:,:,0]), "[test]  get_simulator_batched is not working as expected: times of trivial lumping differ from micro times"
    mean_time_micro = jnp.mean(micro_time_tracker_arrays)
    assert jnp.abs(mean_time_micro-1.0/3) < 1e-1, f"[test]  get_simulator_batched is not working as expected: mean micro time is {mean_time_micro} but should be 1/3"
    paths_meso_states, paths_meso_times = cg.micro_to_meso_batched(micro_state_tracker_arrays,micro_time_tracker_arrays,micro_to_meso_array)
    mean_time_meso = jnp.mean(paths_meso_times)
    #print(f"Mean meso time: {mean_time_meso}")
    assert jnp.abs(mean_time_meso-2.0/3) < 1e-1, f"[test]  get_simulator_batched is not working as expected: mean meso time is {mean_time_meso} but should be 2/3"
    #print(f"Mean micro time: {jnp.mean(micro_time_tracker_arrays)}")
    print("[Test]  passed: _test_simulator_batched")

def _test_all():
    _test_simulator_batched()
    print("[Test]  passed: all simulate.py tests")

if __name__ == "__main__":
    _test_all()