# AR

Simple Association Rules (AR) are a simplified version of the association rule mining technique [Agrawal et al. 1993] with a maximum rule size of two. The method is designed to capture the frequency of two co-occurring events, e.g., “Customers who bought . . . also bought”. 

:::info research paper

[Agrawal et. al., “*Mining Association Rules Between Sets of Items in Large Databases*”. SIGMOD, 1993.](https://dl.acm.org/doi/10.1145/170036.170072)

:::

Algorithmically, the rules and their corresponding importance are “learned” by counting how often the items i and j occurred together in a session of any user. Let a session s be a chronologically ordered tuple of item click events s = ($s_1$,$s_2$,$s_3$, . . . ,$s_m$) and $S_p$ the set of all past sessions. Given a user’s current session s with $s_{|s|}$ being the last item interaction in s, we can define the score for a recommendable item i as follows, where the indicator function $1_{EQ}(a,b)$ is 1 in case a and b refer to the same item and 0 otherwise.

$$
score_{AR}(i,s) = \dfrac{1}{\sum_{p \in S_p}\sum_{x=1}^{|p|}1_{EQ}(s_{|s|},p_x)\cdot(|p|-1)}\sum_{p \in s_p}\sum_{x=1}^{|p|}\sum_{y=1}^{|p|}1_{EQ}(s_{|s|},p_x)\cdot1_{EQ}(i,p_y)
$$

In the above equation, the sums at the right-hand side represent the counting scheme. The term at the left-hand side normalizes the score by the number of total rule occurrences originating from the current item $s_{|s|}$. A list of recommendations returned by the ar method then contains the items with the highest scores in descending order. No minimum support or confidence thresholds are applied.

## References

- [https://arxiv.org/pdf/1803.09587.pdf](https://arxiv.org/pdf/1803.09587.pdf)
- [https://dl.acm.org/doi/10.1145/170036.170072](https://dl.acm.org/doi/10.1145/170036.170072)
- [http://www.rakesh.agrawal-family.com/papers/sigmod93assoc.pdf](http://www.rakesh.agrawal-family.com/papers/sigmod93assoc.pdf)
- [https://github.com/mmaher22/iCV-SBR/tree/master/Source Codes/AR%26SR_Python](https://github.com/mmaher22/iCV-SBR/tree/master/Source%20Codes/AR%26SR_Python)