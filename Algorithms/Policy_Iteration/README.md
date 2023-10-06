# Policy Iteration

Similar to [Value Iteration](../Value_Iteration/README.md), if you find yourself in posession of a fully defined [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process), then you can find the optimal values and the optimal policy in each state. However, now we begin with a policy and improve the policy and our values at each state iteratively. Since there are 1. a finite number of policies, 2. we have the MDP to evaluate how good other actions are not currently in our policy, and 3. we either improve our policy or don't, we converge to the optimal value function, and thus policy.

The algorithm is:

![Policy Iteration Algorithm](./Policy_Iteration_Alg.png)
