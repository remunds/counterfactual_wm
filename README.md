# Goal
- Simple world model based on object-centric representations.
    - This can for example be a transformer-based seq-to-seq model
- Ability to create counterfactual 'dreams' / imagined trajectories.
- Requirements: Use only jax (model and env), s.t. fast experiments are possible 
    - For the model: Use Flax nnx

# Steps
0. Build PPO to familiarize with Flax.nnx 
1. Build seq-2-seq WM that is able to solve simple object-centric games
2. Create ability to create counterfactuals/interventions
