_Sigmoid neurons simulating perceptrons, part I  
Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, c>0c>0. Show that the behaviour of the network doesn't change._

WLOG, choose any perceptron with weights and biases _w_ and _b_ of length _n_. Then, the neuron outputs a 1 iff $w \dot x + b \geq 0$. Implies neuron fires iff $w \dot x \geq b$. Multiplying by a constant c on either side does not change when the neuron fires.

_Sigmoid neurons simulating perceptrons, part II  
Suppose we have the same setup as the last problem - a network of perceptrons. Suppose also that the overall input to the network of perceptrons has been chosen. We won't need the actual input value, we just need the input to have been fixed. Suppose the weights and biases are such that w⋅x+b≠0w⋅x+b≠0 for the input xx to any particular perceptron in the network. Now replace all the perceptrons in the network by sigmoid neurons, and multiply the weights and biases by a positive constant c>0c>0. Show that in the limit as c→∞c→∞ the behaviour of this network of sigmoid neurons is exactly the same as the network of perceptrons. How can this fail when w⋅x+b=0w⋅x+b=0 for one of the perceptrons?_

WLOG Fix x and pick any neuron from a perceptron network. Then that neuron does not fire unless $w \dot x + b \geq 0$. Recall that $\sigma(w, x, b) = \frac{1}{1+ e^(-c(w \dot x + b))}$. Take case 1, $w \dot x + b < 0$. Then, as $ c -> \infinity$, $\sigma -> 0$. If $w \dot x + b > 0$, then $c -> \infinity$ implies $\sigma -> 1$. Note that if $w \dot x + b =0$, then $\sigma = 1/2$ for all values of $c$.

_There is a way of determining the bitwise representation of a digit by adding an extra layer to the three-layer network above. The extra layer converts the output from the previous layer into a binary representation, as illustrated in the figure below. Find a set of weights and biases for the new output layer. Assume that the first 3 layers of neurons are such that the correct output in the third layer (i.e., the old output layer) has activation at least 0.99, and incorrect outputs have activation less than 0.01_

Any bias in (0.04, 0.99) will work for all nodes. If the neurons correspond to bits, then each neuron corresponding to a number will get weights equal to the number's binary 4-bit binary representation across the four nodes. 

For example, X3, which corresponds to $x = 3$, has the 4-bit binary representation 0011, and will get the weight 0 on the nodes corresponding to $2^3$ and $2^4$, and the weight 1 on the nodes corresponding to $2^0$ and $2^1$. 

Therefore, each node will fire iff it is in the binary representation of the relevant digit.
