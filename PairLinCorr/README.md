Here we create a modified version of the problem described in https://github.com/wcchu/NN-expt/tree/master/PairShift.

Assume each item, when picked up each time, generates a tag value `u` to be seen, which can be different from time to time. Assume the true value `v` of the item has a linear correction to the tag value, i.e. `v = c + (1 + d) * u`. If the correction is small, both `c` and `d` are small. The user uses the true values to match two items in a pair, which means

```
c_i + (1 + d_i) * u_i = c_j + (1 + d_j) * u_j
```

or

```
(c_i + d_i * u_i) - (c_j + d_j * u_j) = u_j - u_i
```

which forms the basic equation to solve--`u` values are known tag values and enter into response in the form of `u_j - u_i` and into predictor features as independent values, and `c` and `d` values are tied to each item and what we train in the model.