# Reinforcement learning project

### About
A reinforcement learning agent was trained to visit predefined points in 2D space.  
Project for *Pattern Recognition and Machine Learning* course at *FERIT*

### Requirements
* tf-agents
* numpy
* matplotlib
* scipy

### Usage
run `python train.py` to train the agent.  
run `python test.py` to store all observations in a .csv file for an iteration and display the trajectory using matplotlib.

if you want to render the animation from the csv using blender
* open the blender file
* copy generated .csv file into the render folder
* run the provided script inside the blender text editor
* adjust the length using the timeline
* render using Viewport render animation

### Environment
Actor follows a simplified 2D car/vehicle model. It behaves similarly to the *asteroids* game.  

The environment keeps track of the current actor position, playing area size and other values defining both the actor and the environment itself.  
Playing area of 100x100 m was chosen.  
Targets were generated, using a random uniform distribution. Target positions are in range [-0.25 \* playing_area, 0.25 \* playing area]

### Observations
Environment observations consist of 7 values in range [-1, 1]. 
* values [0] and [1] define the current target location(x, y)
* values [2] and [3] define the current actor location(x, y)
* values [4] and [5] define the current actor velocity vector(x, y)
* value [6] defines the angle difference between current heading and desired heading

### Actions
The agent can choose to apply one of three actions:
* 0 - continue straight
* 1 - turn left (ccw)
* 2 - turn right (cw)

### Rewards
* __(starting angle - ending angle) / max angle__ at each step
* __25 + 75 * (1 - current step / max steps)__ when target is reached
* __-100__ if target area is left
* __25 * (1 - current distance / initial distance)__ at last simulation step, if not done
* __2.5__ some silly edge case...

### Simulation termination
* if the actor leaves the playing area
* if the actor visits the given target
* if the actor fails to rech the target within the defined max_steps

### Results
It sort of works...