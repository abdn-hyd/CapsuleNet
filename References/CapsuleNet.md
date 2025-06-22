# Abstract
#### Definition:
> [!PDF|yellow] [[Capsules.pdf#page=1&selection=14,0,15,75&color=yellow|Capsules, p.1]]
> > A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or an object part.

#### Contribution:
> [!PDF|note] [[Capsules.pdf#page=1&selection=22,27,22,66&color=note|Capsules, p.1]]
> >  recognizing highly overlapping digits.

# Introduction
#### Parse tree structure:
> [!PDF|note] [[Capsules.pdf#page=1&selection=39,45,42,31&color=note|Capsules, p.1]]
> >  for a single fixation, a parse tree is carved out of a fixed multilayer neural network like a sculpture is carved from a rock. Each layer will be divided into many small groups of neurons called “capsules” (Hinton et al. [2011]) and each node in the parse tree will correspond to an active capsule

#### Vector length:
> [!PDF|note] [[Capsules.pdf#page=1&selection=50,22,51,104&color=note|Capsules, p.1]]
> > we explore an interesting alternative which is to use the overall length of the vector of instantiation parameters to represent the existence of the entity and to force the orientation of the vector to represent the properties of the entity. We ensure that the length of the vector output of a capsule cannot exceed 1 by applying a non-linearity that leaves the orientation of the vector unchanged but scales down its magnitude.
> 

#### Output of a specific capsule:
> [!PDF|yellow] [[Capsules.pdf#page=2&selection=11,7,19,103&color=yellow|Capsules, p.2]]
> > Initially, the output is routed to all possible parents but is scaled down by coupling coefficients that sum to 1. For each possible parent, the capsule computes a “prediction vector” by multiplying its own output by a weight matrix. If this prediction vector has a large scalar product with the output of a possible parent, there is top-down feedback which increases the coupling coefficient for that parent and decreasing it for other parents. This increases the contribution that the capsule makes to that parent thus further increasing the scalar product of the capsule’s prediction with the parent’s output.

Using coefficients to determine which parent capsule has the greater contribution to the final output. The study utilize this mechanism to replace "max-pooling" as max-pooling can result in the loss of precise space information.
#### Replacement:
> [!PDF|note] [[Capsules.pdf#page=2&selection=27,79,29,24&color=note|Capsules, p.2]]
> > Even though we are replacing the scalar-output feature detectors of CNNs with vector-output capsules and max-pooling with routing-by-agreemen

# Input and output computation
![[Screenshot 2025-04-22 at 17.16.43.png]]

# Discussion and previous work
