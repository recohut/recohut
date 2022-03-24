# SPop

SPop stands for Session-based Popularity. It is a session popularity predictor that gives higher scores to items with higher number of occurrences in the session. Ties are broken up by adding the popularity score of the item.

The score is given by $r_{s,i} = supp_{s,i} + \frac{supp_i}{(1+supp_i)}$.

## References

- [https://github.com/mmaher22/iCV-SBR/tree/master/Source Codes/S-POP_Python](https://github.com/mmaher22/iCV-SBR/tree/master/Source%20Codes/S-POP_Python)