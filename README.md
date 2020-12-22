### 1.1 Метод ближайшего соседа

Рассмотрим задачу kNN при k = 1 на языке R.

Алгоритм

Для классификации каждого объекта тестовой выборки нужно пошагово произвести ряд следующих операций:

1.Расчёт расстояния до каждого объекта из данной тестовой выборки.

2.Поиск нужного объекта из выборки, от которого растояние до классифицируемого объекта является минимальным.

Класс классифицируемого объекта — это класс, ближайшего к нему объекта из обучающей выборки

3.Введём число точек в тестовой выборке (n). 

4.Теперь создадим n точек с ограничениями по длине и ширине лепестка при помощи cbind и runif. 

5.Отбразим тренировочную выборку. 

6.Рисуя тестовые точки, запускаем алгоритм 1NN для определения принадлежности одному из трёх существующих классов. 

7.В самой функции 1NN ищем ближайшего по Евклидову расстоянию соседа для текущей точки и возвращаем вид ириса для неё же. 

На рисунке ниже показан результат 10 случайно выбранных точек

![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/1nn_10points.png)


Рассмотрим программную реализацию функции 1NN на языке программирования R.
```
oneNN <- function(set, point){
  
  ## возьмём за ближайшего соседа первую точку в наборе
  min_distance <- distance_of_Euclid(set[1, 1:2], point)
  number_of_nearest <- 1
  
  ## попробуем найти соседа ближе
  for(i in 2:N){
    if (distance_of_Euclid(set[i, 1:2], point) < min_distance){
      min_distance <- distance_of_Euclid(set[i, 1:2], point)
      number_of_nearest <- i
    }
  }
  
  ## возвращаем вид ириса ближайшего соседа
  return(set[number_of_nearest, 3])
  }
  ```
  Обратим внимание на карту классификации для 1NN
  
![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/1nn_map.png)

### 1.2 Алгорим k ближайших соседей

Для классифицирования каждого объекта данной выборки требуется пошагово произвести  ряд следующих операций:

1.Вычислить расстояние до каждого из объектов обучающей выборки.

2.Отобрать k объектов обучающей выборки, расстояние до которых минимально.

Класс классифицируемого объекта — это класс, наиболее часто встречающийся среди k ближайших соседей

3.Работа алгоритма k-ближайших соседей начинается с сортировки всех элементов обучающей выборки по расстоянию относительно тестовой точки, далее отправляем в саму функцию kNN обучающую выборку, k и точку, которую нужно классифицировать. 

4.Проходим циклом от 1 до k и смотрим, каких ирисов больше. 

5.Относим классифицируемую точку к тому классу, которого среди этих k элементов больше. 

Ниже можем увидеть код на языке R:
```
## сортировка тренировочного набора
ordered_set <- sort(train_set, points[i, 1:2])
## функция kNN
kNN <- function(k, ordered_arr){
  
  col_class <- dim(ordered_arr)[2]
  
  class <- names(which.max(table(ordered_arr[1:k, col_class])))
  
  return(class)
  
}
```


<p>
  Прежде чем запустить алгоритм kNN для тестовой выборки необходимо выбрать k, для этого будем использовать LOO (leave-one-out CV). Как это выполнить? 
  
  1.На вход мы получаем тренировочную выборку. 
  
  2.Поочерёдно "вытаскиваем" из набора по одной точке и для каждой точки и выборки без неё запускаем kNN, изменяя k от 1 до длины "новой" обучающей выборки. 
  
  3.Сравниваем полученный класс точки с классом этой же точки из изначальной тренировочной выборки, если классы не совпадают, то увеличиваем количество ошибок на данном k на 1. 
  
  4.Пройдя все точки по всем k, ищем минимальное количество ошибок, и индекс этого элемента и будет лучшим k для этой выборки.
</p>

```R
LOO <- function(arr){
  
  row <- dim(arr)[1]
  
  Q <- matrix(0, (row - 1), 1)
  
  for (i in 1:row) {
    
    point <- arr[i, 1:2]
    new_arr <- arr
    new_arr <- new_arr[-i, ]
    ordered_arr <- sort(new_arr, point)
    
    for (k in 1:(row - 1)) {
      
      class <- kNN(k, ordered_arr)
      
      if (class != arr[i, 3]) {
        Q[k] <- Q[k] + 1
      }
      
    }
    
  }
  
  min_k <- which.min(Q[1:(row - 1)])
  min_v <- Q[min_k]
  
  
  I <- matrix(1:(row - 1), (row - 1), 1)
  
  for (i in 1:(row - 1)) {
    Q[i] <- Q[i]/(row - 1)
  }
  
  ## график LOO и k
  plot(
    I[1:(row - 1)], 
    Q[1:(row - 1)], 
    type = "l", xlab = "k", ylab = "LOO",
    main = "LOO(k)"
  )
  points(min_k, min_v/(row - 1), pch = 21, bg = "black")
  
  
  range <- 5
  while (min_k - range < 0 || min_k + range > 149) {
    range <- range - 1
  }
  ## график LOO и k увеличенный масштаб
  plot(
    I[(min_k - range):(min_k + range)], 
    Q[(min_k - range):(min_k + range)], 
    xlim = c((min_k - range), (min_k + range)), 
    ylim = c(min_v/(row - 1) - 0.1, min_v/(row - 1) + 0.1), 
    type = "l", xlab = "k", ylab = "LOO",
    main = "LOO(k) (Окрестность точки)"
  )
  points(min_k, min_v/(row - 1), pch = 21, bg = "black")
  
  return(min_k)
  
}
```
![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/LOO_6nn.png) 

![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/LOO_6nn_near.png)

Для данной выборки LOO возвращает k, равный 6. Теперь запустим 6NN для 10 случаной выбранных точек.

![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/6nn_10points.png)

Посмотрим на карту классификации для 6NN.

![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/6nn_map.png)

### Метод k-ближайших взвешенных соседей

Метод kwNN отличается от kNN тем, что в нём добавляется вес к каждой точке в выборке, потом этот вес суммируется относительно классов, и, по итогу, вес какого класса будет больше, к тому классу и будет отнесена классифицируемая точка. Будем использовать весовую функцию q^i, где q = (0, 1), i = \[1, k\]. Возьмём k = 6 и для него найдём оптимальный q с точностью 0.01.

```
LOO_q <- function(arr, k) {
  
  row <- dim(arr)[1]
  
  Q <- matrix(0, 99, 1)
  
  for (i in 1:row) {
    
    point <- arr[i, 1:2]
    new_arr <- arr
    new_arr <- new_arr[-i, ]
    ordered_arr <- sort(new_arr, point)
    
    weights <- matrix(0, (row - 1), 1)
    
    for (q in 1:99) {
      
      for (p in 1:(row - 1)) {
        weights[p] <- (q / 100)^p
      }
      
      class <- kwNN(k, ordered_arr, weights)
      
      if (class != arr[i, 3]) {
        Q[q] <- Q[q] + 1
      }
      
    }
    
  }
  
  min_q <- which.min(Q[1:99])
  min_v <- min(Q[1:99])
  
  I <- matrix(seq(0.01, 0.99, 0.01), 99, 1)
  
  for (i in 1:99) {
    Q[i] <- Q[i]/100
  }
  
  ## график LOO и q при k = 6
  plot(
    I[1:99], 
    Q[1:99], 
    type = "l", xlab = "q", ylab = "LOO",
    main = "LOO(q) при k = 6"
  )
  points(min_q/100, min_v/100, pch = 21, bg = "black")
  
  ## график LOO и q при k = 6 увеличенный масштаб
  plot(
    I[50:59], 
    Q[50:59], 
    type = "l", xlab = "q", ylab = "LOO",
    main = "LOO(q) при k = 6 (Окрестность точки)"
  )
  points(min_q/100, min_v/100, pch = 21, bg = "black")
  
  return(min_q/100)
  
}
```

![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/LOO_6wnn.png)

![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/LOO_6wnn_near.png)

Сам алгоритм kwNN выглядит так:

```R
kwNN <- function(k, ordered_arr, weights){
  
  order_and_weight <- cbind(ordered_arr, weights)
  classes <- order_and_weight[1:k, 3:4]
  
  w1 <- sum(classes[classes$Species == "setosa", 2])
  w2 <- sum(classes[classes$Species == "versicolor", 2])
  w3 <- sum(classes[classes$Species == "virginica", 2])
  
  answer <- matrix(c(w1, w2, w3), nrow = 1, ncol = 3, byrow = TRUE, list(c(1), c(1, 2, 3)))
  
  class <- c("setosa", "versicolor", "virginica")
  
  return(class[which.max(answer)])
  
}
```

Отобразим 10 случайно выбранных точек с помощью алгоритма kwNN, при k = 6, а q = 0.56.

![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/6wnn_10points.png)

Теперь посмотрим на карту классификации.

![Image alt](https://github.com/Ragnarok7861/Victor/blob/master/6wnn_map.png)
