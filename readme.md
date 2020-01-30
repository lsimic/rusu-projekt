# Reinforcement learning project

### About
A reinforcement learning agent was trained to visit predefined points in 2D space.  
Project for *Pattern Recognition and Machine Learning* course at *FERIT*

### Requirements
* tf-agents
* numpy
* matplotlib
* scipy
* perhaps something else? not sure, gotta check

### Usage
run `python train_v4.py` to train the agent.  
run `python generate_output_v5` to store all observations in a .csv file for an iteration.

### Environment
Actor follows a simplified 2D car/vehicle model. It behaves similarly to the *asteroids* game.  

The environment keeps track of the current actor position, playing area size and other values defining both the actor and the environment itself.  
Playing area of 100x100 m was chosen.  
10 targets were generated, using a random uniform distribution. Target positions are in range [-0.3 \* playing_area, 0.3 \* playing area]

### Observations
Environment observations consist of 8 values in range [-1, 1]. 
* values [0] and [1] define the current target location(x, y)
* values [2] and [3] define the next target location(x, y)
* values [4] and [5] define the current actor location(x, y)
* values [6] and [7] define the current actor velocity vector(x, y)

### Actions
The agent can choose to apply two actions:
* brake or accelerate, value in range [-1, 1]
* turn left or right, value in range [-1, 1]

### Rewards
* __-0.1__ at each step, if the actor moves away from the target point
* __0.1 * movement_distance / max_possible_movement_distance__ at each step, if the actor moves towards the target point
* __5 + 5 * (1 - number_of_steps_taken/number_of_steps_anticipated)__ for each target point reached
* __10 * (max_steps - current_step)/max_steps__ for final target point reached
* __-5__ if actor leaves the playing area
* __5 * (1 - current_distance_to_target / starting_distance_to_target)__ at last simulation step, if not done

### Simulation termination
* if the actor leaves the playing area
* if the actor visits all the points
* if the actor fails to visit any point within 3 * number_of_steps_anticipated

### Results
It likes to drift... a lot