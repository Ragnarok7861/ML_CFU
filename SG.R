L <- function(M) {
  return((M - 1)^2)
}

Xl <- iris[51:150, 3:4]

l <- dim(Xl)[1]
n <- dim(Xl)[2]

classes <- rep(NA, l)

for (i in 1:l) {
  
  if (iris[i + 50, 5] == "versicolor"){
    
    classes[i] <- as.numeric(-1)
    
  }
  if (iris[i + 50, 5] == "virginica"){
  
    classes[i] <- as.numeric(1)
  
  }
  
}

Xl <- cbind(Xl, classes)

# начальный вес
w <- runif(n, -1/(2*n), 1/(2*n))

#лямбда
lambda <- 1/l

#  начальное значение функционала Q
Q <- 0
for (i in 1:l) {
  
  Q <- Q + L(sum(w * Xl[i, 1:n]) * Xl[i, n + 1])
  
}
print(Q)

while (TRUE) {
  
  # выбор случайного объекта
  i <- round(runif(1, 1, 100))
  xi <- Xl[i, 1:n]
  yi <- Xl[i, n + 1]
  
  # ошибка
  eps <- L(sum(w * xi) * yi)
  
  # шаг градиентного спуска
  x <- xi[1]
  y <- xi[2]
  eta <- as.numeric(1 / (sqrt(abs(x)^2 + abs(y)^2)))
  w1 <- w - xi * (2 * eta * (sum(w * xi) - yi))
  
  if ((sum(w * xi) - yi) == 0) {
    break
  }
  
  print(w)
  print(w1)
  
  w <- w1
  
  # пересчёт Q
  Q1 <- (1 - lambda) * Q + lambda * eps
  
  print(Q)
  print(Q1)
  
  if (abs(Q - Q1) < 0.01) {
    break
  }
  
  Q <- Q1
}




color <- c("-1" = "green", "1" = "blue")

print(color[toString(Xl[53, 3])])

plot(
  Xl[, 1], Xl[, 2],
  pch = 21,
  bg = color[(Xl[, 3])],
  col = color[(Xl[, 3])]
)