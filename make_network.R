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
#' @param x data frame of xs for neural network
#' @param y numeric vector of outcomes for network
#'
#' @return formatted data, of class "nn_data"
#' @export
#'
#' @examples
#' dat <- nn_make_data(data.frame(x1 = sample(0:1, 10, replace = T),
#' x2 = sample(0:1, 10, replace = T)), sample(0:1, 10, replace = T))
nn_make_data <- function(x, y){
  # Error Checking
  stopifnot(is.data.frame(x))
  stopifnot(is.numeric(y))
  stopifnot(nrow(x) == length(y))

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
#' nn <- init_network(3,4,4)
#' training <- nn_make_data(data.frame(x1 = sample(0:1, 10, replace = T),
#' x2 = sample(0:1, 10, replace = T),
#' x3 = sample(0:1, 10, replace = T)), sample(1:4, 10, replace = T))
#' test <- nn_make_data(data.frame(x1 = sample(0:1, 10, replace = T),
#' x2 = sample(0:1, 10, replace = T),
#' x3 = sample(0:1, 10, replace = T)), sample(1:4, 10, replace = T))
#'
#' nn <- run_sgd(network = nn,
#' training_data = training,
#' epochs = 3, split_size = 1, eta = 3,
#' test_data = test)
run_sgd <- function(network, training_data, epochs, split_size, eta,
                    test_data = NULL) {
  for (i in seq_len(epochs)) {
    # Randomly assign mini-batches
    n_folds <- ceiling(nrow(training_data)/split_size)
    fold <- sample(1:n_folds, training_data$length, replace = F)

    for (j in seq_len(n_folds)) {
      # Update network for each mini-batch
      mini_batch <- list()
      mini_batch$x <- training_data$x[fold == j,]
      mini_batch$y <- training_data$y[fold == j,]
      network <- update_batch(network, mini_batch, eta)

      # Print diagnostics
      if (!is.null(test_data)) {
        cat(sprintf("Epoch %s: %s/%s",
                    j, nn_eval(test_data), length(test_data)))
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
#' mb <- nn_make_data(x = data.frame(1:6, 7:12), y = runif(6))
#' nn <- update_batch(nn, mb, 3)
update_batch <- function(network, mini_batch, eta) {
  delt <- backprop(network, mini_batch, eta)

  network$weights <- lapply(network$n_layers,
                            function(i) {
                              network$weights[[i]] -
                                (eta/nrow(mini_batch))*delt$weights[[i]]
                            })
  network$biases <- lapply(network$n_layers,
                           function(i) {
                             network$biases[[i]] -
                               (eta/nrow(mini_batch))*delt$biases[[i]]
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
#' mb <- nn_make_data(data.frame(x1 = sample(0:1, 10, replace = T),
#' x2 = sample(0:1, 10, replace = T)), sample(0:1, 10, replace = T))
#' nn <- init_network(c(2, 3, 4))
#' backprop(nn, mb, 3)
backprop <- function(network, mini_batch, eta) {
  # Backpropogate all rows of mini-batch
  bps <- lapply(1:length(mini_batch$y),
                function(i) backprop_single(network,
                                            mini_batch$x[i,],
                                            mini_batch$y[i]))
  bps <- purrr::transpose(bps)

  # Average across rows to change weights
  for (layer in 1:network$n_layers){
    network$weights[[layer]] <- network$weights[[layer]] -
      eta / length(bps$weights) *
      purrr::reduce(bps$weights, function(x, y) x[[layer]] + y[[layer]])
    network$biases[[layer]] <- network$biases[[layer]] -
      eta / length(bps$biases) *
      purrr::reduce(bps$biases, function(x, y) x[[layer]] + y[[layer]])
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
#' d <- nn_make_data(x = data.frame(runif(10), runif(10), runif(10), 
#' runif(10)), y = runif(10))
#' nab <- backprop_single(network, d$x[1,], d$y[1])
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
  nabla$biases[[layers-1]] <- t(cost_derivative(activations[[layers]], y) *
    sigmoid_prime(z[[layers-1]]))
  nabla$weights[[layers-1]] <- t(nabla$biases[[layers-1]]) %*%
    t(activations[[layers-1]])

  # Propogate error through network
  # The indices took FOREVER
  for (layer in seq(layers - 2, 1)){
    sp <- sigmoid_prime(z[[layer]])
    nabla$biases[[layer]] <- nabla$biases[[layer + 1]] %*% 
      network$weights[[layer + 1]] * t(sp)
    
    nabla$weights[[layer]] <- t(nabla$biases[[layer]]) %*%
      activations[[layer]]
  }

  nabla
}


forward_pass <- function(network, x,
                         return = NULL){
  stopifnot(is.null(return) | 1:network$n_layers)

  z <- list()
  activations <- list()
  activations[[1]] <- as.numeric(x)

  for (layer in 1:layers){
    z[[layer]] = network$weights[[layer]] %*% activations[[layer]]
    activations[[layer+1]] <- sigmoid(z[[layer]])
  }

  if (!is.null(return)) {

  }
}

nn_eval <- function(){

}

#' Partial Derivatives of Output wrt final activations
#' @param output_activations activations of final layer
#' @param y actual ys
#'
#'  @return vector of derivatives
#'  @examples
#'  cost_derivative(runif(4), runif(4))
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
