LDF <- function(Py, lambda, n, m, mu, sigma, point) {
  
  point <- as.numeric(point)
  
  p <- rep(0, m)
  
  for (i in 1:m) {
    
    p[i] <- Py[i] * lambda[i]
    
    
    p[i] <- p[i] *  exp(-(1/2) * t(point - mu[i, ]) %*% solve(sigma) %*% (point - mu[i, ])) / sqrt((2 * pi)^n * det(sigma))
    
  }
  
  return(classes[which.max(p)])
  
}


Gauss_distribution <- function(Sigma, mu, x) {
  
  n <- 2
  
  numerator <- exp((-1/2) %*% t(x - mu) %*% solve(Sigma) %*% (x - mu))
  denominator <- sqrt(det(Sigma) * (2 * pi)^n)
  
  #print(numerator)
  #print(denominator)
  
  return(numerator/denominator)
}


set <- iris[ , 3:5]

row <- dim(set)[1]
col <- dim(set)[2]

# количество признаков и количество классов
n <- col - 1 
num_classes <- table(set[n + 1])
m <- dim(num_classes)


classes <- unique(set[, n + 1])
colors <- c("setosa" = "red", "versicolor" = "green", "virginica" = "blue")

# априорная вероятность
Py <- rep(0, m) 

for (i in 1:m) {
  Py[i] <- num_classes[i] / row
}

# "штраф" за ошибку
lambda <- c(1, 1, 1)

# матожидание и дисперсия
mu <- matrix(0, m, n)
sigma <- matrix(0, n, n)

# вычисление матожидания 
for (i in 1:m) {
  for (j in 1:n) {
    
    mu[i, j] <- mean(set[set[, n + 1] == classes[i], ][ , j])
    
  }
}

print(mu)

temp <- rep(0, n)

# вычисление дисперсии

subset <- set[set[, n + 1] == classes[1], ][ , ]
l <- dim(subset)[1]

for (j in 1:l){
  for (i in 1:n) {
    temp[i] <- subset[j, i] - mu[1, i] 
  }
  
  S <- temp %*% t(temp)
  
  for (i1 in 1:n) {
    for (i2 in 1:n) {
      
      sigma[i1, i2] <- sigma[i1, i2] + S[i1, i2]
      
      
    }
  }
}

for (i1 in 1:n) {
  for (i2 in 1:n) {
    
    sigma[i1, i2] <- sigma[i1, i2] / (row - l)
    
  }
}


print(sigma)


plot(
  set[ , 1], set[ , 2], 
  pch = 21, bg = colors[set[, 3]], col = colors[set[, 3]],
  xlab = "Длина лепестка", ylab = "Ширина лепестка",
  main = "Карта классификации (Линейный дискриминант Фишера)",
  xlim = c(1, 7), ylim = c(0, 2.5)
)


# карта классификации
for (i in seq(0.8, 7.2, 0.1)) {
  
  for (j in seq(-0.3, 2.9, 0.1)) {
    
    points(i, j, pch = 1, col = colors[LDF(Py, lambda, n, m, mu, sigma, c(i, j))])
    
  }
  
}


legend(
  "bottomright",
  pch = c(21, 21, 21),
  col = c("red", "green4", "blue"),
  legend = c("setosa", "versicolor", "virginica")
)


Q <- 0

for (i in 1:row) {
  
  point <- set[i, 1:n]
  class <- LDF(Py, lambda, n, m, mu, sigma, point)
  
  if (class != set[i, n + 1]) {
    
    Q <- Q + 1
    
  }
}

Q <- Q / 150

print(Q)


plot(
  set[ , 1], set[ , 2], 
  pch = 21, bg = colors[set[, 3]], col = colors[set[, 3]],
  xlab = "Длина лепестка", ylab = "Ширина лепестка",
  main = "Карта классификации (Линейный дискриминант Фишера)",
  xlim = c(1, 7), ylim = c(0, 2.5)
)

alpha <- solve(sigma) %*% (mu[1,] - mu[2,])
mu_st <- (mu[1,] + mu[2,]) / 2
beta <- mu_st %*% alpha
## Рисуем ЛДФ
abline(beta / alpha[2,1], -alpha[1,1]/alpha[2,1], col = "black", lwd = 1)

alpha <- solve(sigma) %*% (mu[2,] - mu[3,])
mu_st <- (mu[2,] + mu[3,]) / 2
beta <- mu_st %*% alpha
## Рисуем ЛДФ
abline(beta / alpha[2,1], -alpha[1,1]/alpha[2,1], col = "black", lwd = 1)

legend(
  "bottomright",
  pch = c(21, 21, 21),
  col = c("red", "green4", "blue"),
  legend = c("setosa", "versicolor", "virginica")
)
