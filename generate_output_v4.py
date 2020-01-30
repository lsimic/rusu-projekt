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
eval_py_env = EnvironmentV4()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# agent
actor_net = actor_distribution_network.ActorDistributionNetwork(
    eval_env.observation_spec(),
    eval_env.action_spec(),
    fc_layer_params=fc_layer_params
)
optimizer = tensorflow.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
# try to load and overide from the checkpoint...
checkpoint = tensorflow.train.Checkpoint(actor_net=actor_net, optimizer=optimizer)
status = checkpoint.restore(tensorflow.train.latest_checkpoint(CHECKPOINT_DIR))
print(status)
train_step_counter = tensorflow.compat.v2.Variable(0)
agent = reinforce_agent.ReinforceAgent(
    eval_env.time_step_spec(),
    eval_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
agent.initialize()


# dunno why this doesnt work...
print("testing------")
result = []
time_step = eval_py_env.reset()
result.append(time_step.observation)
# print(time_step.is_last())
while not time_step.is_last():
    action_step = agent.policy.action(time_step)
    time_step = eval_py_env.step(action_step.action)
    # print(time_step.is_last())
    result.append(time_step.observation)
with open("trained_outputs.csv","w", newline='') as outfile:
    csv_out=csv.writer(outfile)
    for row in result:
        csv_out.writerow(row)
matplotlib.pyplot.plot([ele[4] for ele in result], [ele[5] for ele in result])
matplotlib.pyplot.scatter([ele[0] for ele in result], [ele[1] for ele in result], c="green")
matplotlib.pyplot.show()
