# Run Network
rm(list = ls())
wd <- "~/Documents/neural-networks-and-deep-learning/"
source(file.path(wd, "load_mnist.R"))
source(file.path(wd, "network_1.R"))

load_mnist(file.path(wd, "data", "mnist"))
nn <- init_network(c(ncol(train$x), 30, 10))
train <- nn_make_data(nn, as.data.frame(train$x[1:5000,]), train$y[1:5000])
test <- nn_make_data(nn, as.data.frame(test$x[1:200, ]), test$y[1:200])

nn <- run_sgd(nn, train, 3, 10, 3, test)
