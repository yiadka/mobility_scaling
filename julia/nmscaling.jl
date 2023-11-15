# sample data from beta distribution
using Distributions
using Plots
# using StatPlots

d = Normal(171, 6)
println( cdf(d, 175) - cdf(d, 165) )
Plots.plot(d, fill=(0, .5,:orange))