library(keras)

d <- dataset_mnist()

train_x <- d$train$x
test_x <- d$test$x
train_y <- d$train$y
test_y <- d$test$y

# Convert each 2D image matrix into 1D input vector and normalize
norm <- function(x) {
  (array(x, dim = c(dim(x)[1], 
                   prod(dim(x)[-1]))) - mean(x)) / 
    (max(x) - mean(x))
}

train_x <- norm(train_x)
test_x <- norm(test_x)
  
# dummy out Y
train_y <- to_categorical(train_y, 10)
test_y <- to_categorical(test_y, 10)

# Define a sequential model
model <- keras_model_sequential()

# Define a model with 1 input layer, 1 hidden layer w/ dropout @ 0.4, and output layer with digits 0-9
model %>% 
  layer_dense(units = 784, input_shape = 784) %>%
  layer_dropout(0.4) %>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 10) %>%
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy', 
  optimizer = 'adam', 
  metrics = 'accuracy'
)

#fitting the model on the training dataset
model %>% fit(train_x, train_y, epochs = 3, batch_size = 128)

#Evaluating model on the cross validation dataset
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)
