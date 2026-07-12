# Foundations: Non-Convex and Multi-Objective Optimization

## Convex versus non-convex

For a convex objective, the line between any two points on the function lies above the function. Consequently, every local minimum is global. Deep networks combine nonlinear activations, layers, attention, normalization, and parameter symmetries, producing saddle points, flat regions, sharp valleys, and many basins.

## Dominance and the Pareto set

For minimization, candidate `a` dominates `b` when `a` is no worse in every objective and strictly better in at least one. A candidate is non-dominated when no other candidate dominates it. The image of all non-dominated decisions in objective space is the Pareto frontier.

## Gradient conflict

For objectives `Li` and `Lj`, gradients conflict locally when their inner product is negative:

```text
grad(Li) dot grad(Lj) < 0
```

An update that helps one task may then hurt the other. Weighted sums select a compromise direction; gradient-surgery methods can project conflicting components; multi-gradient approaches search for a common descent direction.

## Why scalarization can miss solutions

A weighted sum traces points supported by a tangent hyperplane. When the attainable objective region is non-convex, portions of the Pareto frontier may not be optimal for any positive linear weighting. Constraint methods, evolutionary search, or direct Pareto-set learning can expose more of the frontier.

## Three distinct uses of neural networks

1. **The object being optimized:** training network parameters against several losses.
2. **A learned optimizer:** proposing parameter updates or experiments from optimization history.
3. **A solution generator:** producing decisions conditioned on an objective-preference vector.

Keeping these roles separate prevents the common mistake of claiming that an architecture itself “solves” multi-objective optimization.

