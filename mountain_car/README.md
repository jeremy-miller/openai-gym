# Mountain Car v0
The Mountain Car environment is described as follows on the OpenAI Gym website:

> A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the
> mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single
> pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.
> 
> [Source](https://gym.openai.com/envs/MountainCar-v0/)

Further information related to the Mountain Car environment, as paraphrased from
[this](https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f)
blog post:

The car’s state, at any point in time, is given by a vector containing its horizontal position and velocity.
The car commences each episode stationary, at the bottom of the valley between the mountains (at position
approximately -0.5), and the episode ends when either the car reaches the flag (position > 0.5) or after 200
moves. At each move, the car has three actions available to it: push left, push right or do nothing. A penalty
of 1 unit is applied for each move taken (including doing nothing).

This same blog post is used as inspiration for my implementation.
