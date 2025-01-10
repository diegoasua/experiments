# Rationale

This is a brief explanation of why Hutchinson works. For a random vector 𝓋 with mean 0 and covariance I it holds that
$$\mathbb{E}[\mathbf{v}^\top \mathbf{A} \mathbf{v}] = \text{tr}(\mathbf{A})$$

# Proof

To demonstrate  $\mathbb{E}[\mathbf{v}^\top \mathbf{A} \mathbf{v}] = \text{tr}(\mathbf{A})$ , where  $\mathbf{v}$  is a random vector with independent standard normal entries and  $\mathbf{A}$  is a symmetric matrix:

1. Expand the expectation:

$$\mathbb{E}[\mathbf{v}^\top \mathbf{A} \mathbf{v}] = \mathbb{E}\left[\sum_{i,j} v_i A_{ij} v_j\right].
$$

2. Separate diagonal and off-diagonal terms:

$$\mathbb{E}\left[\sum_{i,j} v_i A_{ij} v_j\right] = \sum_{i} A_{ii} \mathbb{E}[v_i^2] + \sum_{i \neq j} A_{ij} \mathbb{E}[v_i v_j].$$

3.	Use properties of  v_i :
- variance of a standard normal $ \mathbb{E}[v_i^2] = 1$
- independence of entries $\mathbb{E}[v_i v_j] = 0$  for  $i \neq j$ 

4.	Simplify:

$$\mathbb{E}[\mathbf{v}^\top \mathbf{A} \mathbf{v}] = \sum_{i} A_{ii} = \text{tr}(\mathbf{A}).$$

Extra: this is possible because:
-  $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
- and for independent observations $\mathbb{E}[v_i v_j] = \mathbb{E}[v_i] \cdot \mathbb{E}[v_j].$
- and for standard normal random variables, their mean is $\mathbb{E}[v_i] = \mathbb{E}[v_j] = 0 .$