# SR

SR stands for Sequential Rules. The SR method is a variation of MC and AR. It also takes the order of actions into account, but in a less restrictive manner. In contrast to the MC method, we create a rule when an item q appeared after an item p in a session even when other events happened between p and q. When assigning weights to the rules, we consider the number of elements appearing between p and q in the session.

:::info research paper

[Kamehkhosh et. al., “A Comparison of Frequent Pattern Techniques and a Deep Learning Method for Session-Based Recommendation”. RecSys, 2017.](http://ceur-ws.org/Vol-1922/paper10.pdf)

:::

Specifically, we use the weight function $w_{SR}(x)$ = 1/(x), where x corresponds to the number of steps between the two items. Given the current session s, the sr method calculates the score for the target item i as follows:

$$
score_{SR}(i,s) = \dfrac{1}{\sum_{p \in S_p}\sum_{x=2}^{|p|}1_{EQ}(s_{|s|},p_x)\cdot x}\sum_{p \in s_p}\sum_{x=2}^{|p|}\sum_{y=1}^{x-1}1_{EQ}(s_{|s|},p_y)\cdot1_{EQ}(i,p_x)\cdot w_{SR}(x-y)
$$

In contrast to the equation for AR, the third inner sum only considers indices of previous item view events for each session p. In addition, the weighting function $w_{SR}(x)$ is added. Again, we normalize the absolute score by the total number of rule occurrences for the current item $s_{|s|}$.

## References

- [https://arxiv.org/pdf/1803.09587.pdf](https://arxiv.org/pdf/1803.09587.pdf)
- [http://ceur-ws.org/Vol-1922/paper10.pdf](http://ceur-ws.org/Vol-1922/paper10.pdf)
- [https://github.com/mmaher22/iCV-SBR/tree/master/Source Codes/AR%26SR_Python](https://github.com/mmaher22/iCV-SBR/tree/master/Source%20Codes/AR%26SR_Python)