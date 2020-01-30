import tensorflow
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, actor_distribution_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
tensorflow.compat.v1.enable_v2_behavior()

import matplotlib.pyplot
import csv
import os

from environment_v4 import EnvironmentV4


CHECKPOINT_DIR = "checkpoints_v4"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_episode(environment, policy, num_episodes):
    episode_counter = 0
    environment.reset()
    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)
        if traj.is_boundary():
            episode_counter += 1

def create_output_csv(policy, filename):
    result = []
    time_step = eval_env.reset()
    result.append(time_step.observation.numpy())
    print(time_step.is_last())
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        print(time_step.is_last())
        result.append(time_step.observation.numpy())
    with open(filename,"w", newline='') as outfile:
        csv_out=csv.writer(outfile)
        for row in result:
            csv_out.writerow(row)

# hyperparameters
num_iterations = 5001
collect_episodes_per_iteration = 5
replay_buffer_capacity = 10000
learning_rate = 1e-3
log_interval = 50
num_eval_episodes = 10
eval_interval = 100
fc_layer_params = (32,32)
checkpoint_interval = 1000

# environments
train_py_env = EnvironmentV4()
eval_py_env = EnvironmentV4()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# agent
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)
optimizer = tensorflow.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
# try to load and overide from the checkpoint...
checkpoint = tensorflow.train.Checkpoint(actor_net=actor_net, optimizer=optimizer)
status = checkpoint.restore(tensorflow.train.latest_checkpoint(CHECKPOINT_DIR))
print(status)
train_step_counter = tensorflow.compat.v2.Variable(0)
agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
agent.initialize()

# replay buffer, dataset
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity
)

# train the agent
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)
# Reset the train step
agent.train_step_counter.assign(0)
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_env, agent.collect_policy, collect_episodes_per_iteration)
    # Use data from the buffer and update the agent's network.
    experience = replay_buffer.gather_all()
    train_loss = agent.train(experience)
    replay_buffer.clear()

    step = agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
    if step % checkpoint_interval == 0:
        checkpoint = tensorflow.train.Checkpoint(actor_net=actor_net, optimizer=optimizer)
        checkpoint.save(file_prefix=CHECKPOINT_PREFIX)


# plot...
iterations = range(0, num_iterations + 1, eval_interval)
matplotlib.pyplot.plot(iterations, returns)
matplotlib.pyplot.ylabel('Average Return')
matplotlib.pyplot.xlabel('Iterations')
matplotlib.pyplot.ylim(top=3)
matplotlib.pyplot.show()
