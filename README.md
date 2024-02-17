#### !!! Disclaimer !!!!

This is a personal project that I worked on as an learning excercise.
Please, do not use it as an example of any kind!

The code herein represents my very first steps into the Rust language.
It is likely full of missed opportunities to do things the idiomatic Rust way!  
If I were to continue iterating on it, the internal organization of the various types would be my first priority.  
Right now, I need to move on to other things.

## What?

This is a version of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) implemented in Rust, as a learning excercise.

## Why?

I've been meaning to try out Rust for a while. Now that I have some time, I needed a project.
Micrograd seemed simple enough to be a good fit for the purpose.

## How?

At the core of micrograd is the `Value` class.
A computational graph, and by extension a NN, is a graph of `Value` objects.
A `Value` stores a scalar value, a gradient, and references to its inputs.
It also "knows" how to back-propagate the gradient to those inputs.

Since the computational graph is built by performing operations on values, it quickly became apparent that in Rust those values will need to be behind an `Rc`.
Passing Rcs around didn't _feel_ right, so I wrapped my version of `Value` in a `Node` struct, which holds the Rc, along with a few relevant operations.
`Node`s are cheap and easy to clone, since they are just thin wrappers.
For that reason, math operator overloads are defined on `Node` and not `Value`.

In micrograd, backprop for different operations is expressed as lambda functions.
I could have taken that approach too, but I decided to define it as a trait.
This way the set of operators for which backprop is defined can be extended independently / externally of the Value struct.

The current set of supported operators is represented by the `Expr` enum.

## Notes

I have included an extra `Sum` operation that uses a single node to represent multiple additions.
This allows for a substatial reduction in the total number of nodes in some cases.

In particular, the demo MLP with 2 inputs, two layers of 16 neurons each, and one output neuron (337 total parameters), plus the loss function results in ~72000 nodes without the Sum expression.
With it, the total number of nodes in the graph goes down to ~41000.

In the python version, on my dev machine, the total runtime for 100 iterations over the full training data set of 100 examples is ~2m31s.
Making use of the Sum operation brings that down to 1m34s.

The Rust version is much faster, as expected, taking on average 6.22s without the Sum optimization, and 4.8s with it.
Even that is, of course, laughably slow in the context of NN training, given how small the neural network is.
Performance is not what micrograd is about. :)
