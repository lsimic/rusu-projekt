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

from environment_test import TestEnvironment


CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")


# hyperparameters
learning_rate = 1e-3
fc_layer_params = (32,32)

# environments
eval_py_env = TestEnvironment()
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

print("testing------")
result = []
time_step = eval_env.reset()
result.append(time_step.observation[0].numpy())

while not time_step.is_last():
    action_step = agent.policy.action(time_step)
    time_step = eval_env.step(action_step.action)
    result.append(time_step.observation[0].numpy())

with open("trained_outputs.csv","w", newline='') as outfile:
    csv_out=csv.writer(outfile)
    for row in result:
        csv_out.writerow(row)

targets = []
counter=1
for i in range(0, len(result)-1):
    if not result[i][0] == result[i+1][0]:
        targets.append((result[i][0], result[i][1], counter))
        counter += 1
targets.append((result[i][0], result[i][1], counter))

print(len(targets))

matplotlib.pyplot.plot([ele[2] for ele in result], [ele[3] for ele in result])
matplotlib.pyplot.scatter([result[0][2]], [result[0][3]], color="g")
fig = matplotlib.pyplot.gcf()
ax = fig.gca()
for item in targets:
    circle = matplotlib.pyplot.Circle((item[0], item[1]), 0.1, color="g", fill=False)
    ax.add_artist(circle)
    matplotlib.pyplot.text(item[0], item[1], str(item[2]), color="g", fontsize=12)

matplotlib.pyplot.ylim(bottom=-1.0, top=1.0)
matplotlib.pyplot.xlim(left=-1.0, right=1.0)
matplotlib.pyplot.show()
