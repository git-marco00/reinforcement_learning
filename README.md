# REINFORCEMENT LEARNING
This project was created as part of the Symbolic and Evolutionary Artificial Intelligence course at the University of Pisa (2023-2024).

## What's Inside
- **Policy Gradient Methods**: Focused on optimizing the policy directly.

## Algorithms
1. **Q-learning**: Simple state-action value function.
2. **REINFORCE**:
   - Vanilla: Basic approach with return Gt.
   - With Baseline: Adds a value function to reduce variance.
3. **Actor-Critic**:
   - Vanilla: Combines actor-only and critic-only methods.
   - A2C: Uses advantage function for stability.
   - A2C with Target Networks: Adds stability with target networks.
4. **PPO**: Proximal Policy Optimization - easy, stable, and efficient.

## Benchmark
The chosen benchmark is the CartPole-v1 environment to test the algorithms. The goal is to balance a pole on a moving cart.

## Results
I compared how fast each algorithm converges. Spoiler: PPO is the fastest, A2C with target networks requires the fewest episodes, and REINFORCE with an advantage function needs the fewest optimization steps.
