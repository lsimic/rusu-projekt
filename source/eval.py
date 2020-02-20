import tensorflow
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
tensorflow.compat.v1.enable_v2_behavior()

import matplotlib.pyplot
import numpy
import os
import csv
import statistics

from environment_test import TestEnvironment


CHECKPOINT_DIR = "checkpoints_5k"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")


# hyperparameters
learning_rate = 1e-3
fc_layer_params = (32,32)
target_count = 1

# environments
eval_py_env = TestEnvironment()
eval_py_env.max_target_count = target_count
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# agent
q_net = q_network.QNetwork(
    eval_env.observation_spec(),
    eval_env.action_spec(),
    fc_layer_params=fc_layer_params
)
optimizer = tensorflow.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

# try to load and overide from the checkpoint...
checkpoint = tensorflow.train.Checkpoint(q_net=q_net, optimizer=optimizer)
status = checkpoint.restore(tensorflow.train.latest_checkpoint(CHECKPOINT_DIR))
print(status)

train_step_counter = tensorflow.compat.v2.Variable(0)
agent = dqn_agent.DqnAgent(
    eval_env.time_step_spec(),
    eval_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()

success_count = 0
resulting_times = []
for i in range(0, 100):
    result = []
    time_step = eval_env.reset()
    result.append(time_step.observation[0].numpy())

    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        result.append(time_step.observation[0].numpy())

    targets = []
    counter=1
    for i in range(0, len(result)-1):
        if not result[i][0] == result[i+1][0]:
            targets.append((result[i][0], result[i][1], counter))
            counter += 1
    targets.append((result[i][0], result[i][1], counter))

    if len(targets) >= target_count:
        success_count += 1
        resulting_times.append(len(result)*eval_py_env.time_step)

print("success: {0}".format(success_count))
if success_count > 2:
    print("average: {0}".format(statistics.mean(resulting_times)))
    print("stdev: {0}".format(statistics.stdev(resulting_times)))
