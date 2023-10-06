# On-Policy Monte Carlo

Imagine that you do not have access to a Markov Decision Process. The Policy Iteration and Value Iteration Algorithms are no longer possible because you do not have access to the underlying transition dynamics. A consequence of this is that you can no longer store only a greedy policy. We relied previously on the dynamics to update the values of states unreachable by our policy. Now, we can only rely on our policy to interact with the world and gather information about it. So, we must keep a stochastic policy that explores occasionally so we can continually update our action values.

The MDP being inaccessible does not remove the fact that there is a unique optimal state-action value map for each state-action pair, and therefore a certain family of equivalently optimal greedy policies. The first-visit updates in this algorithm creates an unbiased estimate of the state-action value map and thus after infinite experience with a stochastic policy will converge to the optimal state-action value map. Since $\epsilon > 0$ our policy will remain stochastic, but the collection of most probable actions in each state for our policy derived after infinite iterations of the algorithm will equal one of the optimal greedy policies.  

The algorithm is:  

![On-Policy First-visit Monte Carlo](./MC_On_FV_Soft_Alg.png)