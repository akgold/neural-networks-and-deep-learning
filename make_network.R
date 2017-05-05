
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

  nn <- lapply(2:length(sizes),
               function(i){
                 l <- init_layer(size = sizes[i],
                                 prev_layer_nodes = sizes[i-1])
                 l$layer <- i
                 l
               })
  nn <- data.table::rbindlist(nn)


  class(nn) <- append(class(nn), "nn")

  nn
}



#' Create a layer of a neural network
#'
#' @param size number of neurons in this layer
#' @param prev_layer_nodes number of neurons in the prior layer
#'
#' @return
#' @export
#'
#' @examples
#' init_layer(10, 4)
init_layer <- function(size, prev_layer_nodes){
  l <- data.table::rbindlist(
    lapply(1:size,
           function(i) init_neuron(prev_layer_nodes)))
  l[, size := size]
  l[, prev_node := 1:prev_layer_nodes]

  l
}

#' Create a single Neuron
#'
#' @param input_nodes integer number of input nodes
#'
#' @return a data.table with `input_nodes` unique weights and one bias
#' @export
#'
#' @examples
#' init_neuron(8)
init_neuron <- function(input_nodes){
  data.table(bias = rnorm(1),
             weights = rnorm(input_nodes))
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

  # Format
  l <- data.frame(cbind(x, y))
  names(l) <- NULL
  names(l)[ncol(l)] <- "y"
  class(l) <- append(class(l), "nn_data")

  l
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
  stopifnot("nn_data" %in% class(training_data))
  stopifnot(is.null(test_data) | "nn_data" %in% class(test_data))

  for (i in 1:epochs) {
    # Randomly assign mini-batches
    n_folds <- ceiling(nrow(training_data)/split_size)
    fold <- sample(1:n_folds, training_data$length, replace = F)

    for (j in 1:n_folds) {
      # Update network for each mini-batch
      mini_batch <- training_data[fold == j,]
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
#' @return
#' @export
#'
#' @examples
update_batch <- function(network, mini_batch, eta) {
  delt <- backprop(network, mini_batch)

  network$weights <- network$weights - (eta/nrow(mini_batch))*delt$weights
  network$biases <- network$biases - (eta/nrow(mini_batch))*delt$biases
}

#UNFINISHED
backprop <- function(network, x, y) {

  network <- data.table::rbindlist(list(network,
                                   data.table(z = NA,
                                              activations = as.numeric(x),
                                              layer = 1)),
                                   use.names = T, fill = T)
  activations <- lapply(2:max(network$layer),
                      function(layer)
                        get_activations(activations, network, layer))

}

get_activations <- function(network, layer){
  prev_a <- network[network$layer == layer-1]$activations
  network[, z := weights * prev_a + bias]
  network[, activations := sigmoid(z)]
  layer
}


nn_eval <- function(){

}

feed_forward <- function(network, dat) {
  sigmoid(network$weights %*% dat + network$biases)
}


#' Compute sigmoid function of argument
#'
#' @param z the function
#'
#' @return sigmoid(z)
#' @export
#'
#' @examples
#' sigmoid(3.5676)
sigmoid <- function(z) {
  1/(1 + exp(-z))
}

