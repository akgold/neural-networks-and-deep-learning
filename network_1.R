#' Initialize a neural network with biases and weights
#'
#' Both biases and weights are `~N(0,1)`.
#'
#' @param sizes a numeric vector of the number of neurons in each level of the network
#'
#' @return A neural network, named list
#' @export
#'
#' @examples
#' nn <- init_network(sizes = c(4,10,6))
init_network <- function(sizes) {
  stopifnot(is.numeric(sizes))
  stopifnot(all(sizes > 0))
  stopifnot(length(sizes) > 1)


  # Initialize weights and biases
  list(weights = lapply(1:(length(sizes)-1),
                        function(i) matrix(rnorm(sizes[i]*sizes[i+1]),
                                           ncol = sizes[i])),
       biases = lapply(sizes[2:length(sizes)],
                       function(size) rnorm(size)),
       sizes = sizes,
       n_layers = length(sizes))
}

#' Format Input and Output Data for Neural Network
#'
#' Note that the output data, y, should be a data frame with the same number 
#' of columns as the network's output layer.
#'
#' @param x data frame of xs for neural network
#' @param y data frame of ys for neural network
#'
#' @return formatted data, of class "nn_data"
#' @export
#'
#' @examples
#' nn <- init_network(c(2,4,2))
#' nn_make_data(nn, data.frame(sample(0:1, 10, replace = T),
#' sample(0:1, 10, replace = T)), sample(1:2, 10, replace = T))
nn_make_data <- function(network, x, y){
  # Error Checking
  stopifnot(is.data.frame(x) & is.numeric(y))
  stopifnot(nrow(x) == nrow(y))
  stopifnot(ncol(x) == network$sizes[1])
  stopifnot(max(y) <= network$sizes[length(network$sizes)])
  
  # Turn categorical Y into vector
  y <- factor(y, levels = seq(network$sizes[length(network$sizes)]))
  y <- as.data.frame(model.matrix(~ y - 1))
  
  list(x = x, y = y)
}

#' Train a network using SGD
#'
#' @param network a neural network
#' @param epochs number of epochs to run
#' @param split_size number of observations in each mini-batch
#' @param eta step size for gradient descent
#' @param training_data training data, as formatted by ?nn_make_data
#' @param test_data test data, as formatted by ?nn_make_data
#'
#' @return the network, with weights and biases tuned using training data
#' @export
#'
#' @examples
#' nn <- init_network(c(3,4,4))
#' training <- nn_make_data(nn, data.frame(runif(10), runif(10), runif(10)), 
#' sample(1:4, 10, replace = T))
#' test <- nn_make_data(nn, data.frame(runif(10), runif(10), runif(10)), 
#' sample(1:4, 10, replace = T))
#'
#' nn <- run_sgd(network = nn, training_data = training, epochs = 3, 
#' split_size = 1, eta = 3, test_data = test)
run_sgd <- function(network, training_data, epochs, split_size, eta,
                    test_data = NULL) {
  for (i in seq_len(epochs)) {
    # Randomly assign mini-batches
    n_folds <- ceiling(nrow(training_data$x)/split_size)
    fold <- sample(1:n_folds, nrow(training_data$x), replace = F)

    for (j in seq_len(n_folds)) {      
      print(j)
      print(network)
      # Update network for each mini-batch
      mini_batch <- list()
      mini_batch$x <- training_data$x[fold == j,]
      mini_batch$y <- training_data$y[fold == j,]
      network <- update_batch(network, mini_batch, eta)


      # Print diagnostics
      if (!is.null(test_data)) {
        cat(sprintf("Epoch %s: %s/%s",
                    j, nn_eval(nn, test_data), nrow(test_data$x)))
      } else {
        sprintf("Epoch %s: ", j)
      }
    }
  }
  network
}


#' Update Mini-Batch Parameters
#'
#' Update the network's weights and biases by applying
#' gradient descent using backpropagation to a single mini batch.
#'
#' @inheritParams run_sgd
#'
#' @return network with weights and biases updated according to 1 mini batch
#' @export
#'
#' @examples
#' nn <- init_network(c(2, 4, 3))
#' mb <- nn_make_data(nn, x = data.frame(1:5, 6:10), 
#'     y = sample(1:3, 10, replace = T))
#' nn <- update_batch(nn, mb, 3)
#' nn
update_batch <- function(network, mini_batch, eta) {
  delt <- backprop(network, mini_batch, eta)
  n <- nrow(mini_batch$x)

  network$weights <- lapply(seq_len(network$n_layers - 1),
                            function(i) {
                              network$weights[[i]] -
                                eta/n*delt$weights[[i]]
                            })
  network$biases <- lapply(seq_len(network$n_layers - 1),
                           function(i) {
                             network$biases[[i]] -
                               eta/n*delt$biases[[i]]
                           })
  network
}


#' Backpropogate errors across network for a mini-batch
#'
#'
#' @param network
#' @param mini_batch
#' @param eta
#'
#' @examples
#' nn <- init_network(c(2, 3, 4))
#' mb <- nn_make_data(nn, data.frame(runif(10), runif(10)), 
#' sample(1:4, 10, replace = T))
#' backprop(nn, mb, 3)
backprop <- function(network, mini_batch, eta) {
  n <- nrow(mini_batch$x)
  
  # Backpropogate all rows of mini-batch
  bps <- lapply(seq(n),
                function(i) backprop_single(network,
                                            mini_batch$x[i,],
                                            mini_batch$y[i,]))
  
  # Average across runs of backprops to update network
  for (layer in seq(network$n_layers-1)) {
    w <- network$weights[[layer]]
    b <- network$biases[[layer]]
    
    # Reduce all delta weights into network
    for(i in seq(n)){
      w <- w - eta / n * bps[[i]]$weights[[layer]]
      b <- b - eta / n * bps[[i]]$biases[[layer]]
    }
    network$weights[[layer]] <- w
    network$biases[[layer]] <- b
  }

  network
}

#' Backpropogate error based on a single input to the network
#' @param network a neural network as configured by init_network
#' @param x a vector of xs
#' @param y a vector of ys
#'
#' @return the gradient for the cost function wrt each bias and weight
#'
#' @examples
#' network <- init_network(c(4, 3, 2))
#' d <- nn_make_data(network, x = data.frame(runif(10), runif(10), runif(10), 
#' runif(10)), y = sample(1:2, 10, replace = T))
#' nab <- backprop_single(network, d$x[1,], d$y[1,])
#' nab
backprop_single <- function(network, x, y){
  layers <- network$n_layers
  stopifnot(ncol(network$weights[[1]]) == length(x))

  # Set up output
  z <- list()
  activations <- list()
  activations[[1]] <- as.numeric(x)

  # Do forward pass
  for (layer in seq_len(layers-1)){
    z[[layer]] = network$weights[[layer]] %*% activations[[layer]] +
      network$biases[[layer]]
    activations[[layer + 1]] <- sigmoid(z[[layer]])
  }

  # Set up nablas for each layer
  nabla <- list()
  nabla$weights <- list()
  nabla$biases <- list()

  # Get Nablas for output layer
  nabla$biases[[layers-1]] <- as.numeric(cost_derivative(activations[[layers]], y) *
    sigmoid_prime(z[[layers-1]]))
  # Why the heck do I need t(t(.))??
  nabla$weights[[layers-1]] <- nabla$biases[[layers-1]] %*% 
    t(activations[[layers-1]]) 
    
  
  # Propogate error through network
  for (layer in seq(layers - 2, 1)){
    sp <- sigmoid_prime(z[[layer]])
    nabla$biases[[layer]] <- as.numeric(nabla$biases[[layer + 1]] %*% 
      network$weights[[layer + 1]] * t(sp))
    
    nabla$weights[[layer]] <- t(t(nabla$biases[[layer]])) %*%
      activations[[layer]]
  }
  
  nabla
}

#' Partial Derivatives of Output wrt final activations
#' @param output_activations activations of final layer
#' @param y vector of actual ys
#'
#'  @return vector of derivatives
#'  @examples
#'  cost_derivative(runif(4), c(0,0,1,0))
cost_derivative <- function(output_activations, y){
  output_activations - y
}

#' Compute sigmoid function of argument
#'
#' @param z the function
#'
#' @return sigmoid(z)
#' @export
#'
#' @examples
#' sigmoid(runif(5))
sigmoid <- function(z) {
  1/(1 + exp(-z))
}

#' Compute derivative sigmoid function of argument
#'
#' @param z the function
#'
#' @return derivative of sigmoid at z
#' @export
#'
#' @examples
#' sigmoid_prime(runif(5))
sigmoid_prime <- function(z) {
  sigmoid(z) * (1 - sigmoid(z))
}

#' Evaluate Performance of network
#'
#' @param nn a neural network created by ?init_network
#' @param test_data test data as created by ?nn_make_data
#'
#' @return number of outputs the network gets right
#'
#' @examples
#' nn <- init_network(c(3,4,5))
#' d <- nn_make_data(nn, data.frame(runif(10), runif(10), runif(10)), 
#' sample(1:5, 10, replace = T))
#' nn_eval(nn, d)
#' 
nn_eval <- function(nn, test_data){
  # Get output of network as vector
  outs <- feedforward(nn, test_data$x)
  
  # Get single output as argmax of y
  out_num <- apply(outs, 1, function(y) which(y == max(y)))
  # Do same for actual test data
  y_num <- apply(test_data$y, 1, function(y) which(y == max(y)))
  
  sum(out_num == y_num)
}


#' Get output of network for given set of Xs
#'
#' @param nn a neural network
#' @param x a data.frame of Xs
#' 
#' @return the output of the output layer of network given X
#' 
#' @example 
#' nn <- init_network(c(3,4,5))
#' d <- nn_make_data(nn, data.frame(runif(10), runif(10), runif(10)), 
#' sample(1:5, 10, replace = T))
#' feedforward(nn, d$x)
feedforward <- function(nn, x){
  
  # Rows of x to list
  x <- split(x, seq(nrow(x)))
  
  # Loop over x vectors
  o <- lapply(x, function(act){
    act <- as.numeric(act)
    # Feed forward each input activation
    for (layer in seq_len(nn$n_layers-1)){
      z <- nn$weights[[layer]] %*% act +
        nn$biases[[layer]]
      act <- sigmoid(z)
    }
    t(act)
  })
  
  as.data.frame(do.call(rbind, o))
}