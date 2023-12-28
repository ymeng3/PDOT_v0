
# Each setting will be run 10 times. Average behavior and confidence band are reported.
# m = n in 100, 300, 1000, 3000, 10000
# kkt tolerance = 1e-4
# comparison: Sinkhorn in POT

using Random, Distributions, Distances

d = 2
m, n = 100, 100

ran_seed = 123
Random.seed!(ran_seed)
rng = Random.MersenneTwister(ran_seed)
mu_s = rand(rng, Normal(0.0,1.0), 2)
A_s = rand(rng, Normal(0.0,1.0), (2,2))
cov_s = A_s * A_s'
x_s = rand(rng, MvNormal(mu_s, cov_s), m)

ran_seed = 321
Random.seed!(ran_seed)
rng = Random.MersenneTwister(ran_seed)
mu_t = rand(rng, Normal(5.0,10.0), 2)
A_t = rand(rng, Normal(0.0,1.0), (2,2))
cov_t = A_t * A_t'
x_t = rand(rng, MvNormal(mu_t, cov_t), n)

cost_matrix = pairwise(Euclidean(), x_s, x_t, dims=2).^2

source_distribution = ones(m)./m 
target_distribution = ones(n)./n