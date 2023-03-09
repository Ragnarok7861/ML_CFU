# Задачи классификации

## Оглавление
1. [Метрические алгоритмы классификации](#Метрические-алгоритмы-классификации)
    1. [Метод ближайшего соседа](#Метод-ближайшего-соседа)
    1. [Метод k-ближайших соседей](#Метод-k-ближайших-соседей)
    1. [Метод k-ближайших взвешенных соседей](#Метод-k-ближайших-взвешенных-соседей)
    1. [Преимущества kwNN](#Преимущества-kwNN)
    1. [Метод парзеновского окна](#Метод-парзеновского-окна)
    1. [Метод потенциальных функций](#Метод-потенциальных-функций)
2. [Байесовские методы классификации](#Байесовские-методы-классификации)
    1. [Линии уровня нормального распределения](#Линии-уровня-нормального-распределения)
    2. [Наивный нормальный байесовский классификатор](#Наивный-нормальный-байесовский-классификатор)
    3. [Подстановочный алгоритм](#Подстановочный-алгоритм)
    4. [Линейный дискриминант Фишера](#Линейный-дискриминант-Фишера)
3. [Линейные методы классификации](#Линейные-методы-классификации)
    1. [Стохастический градиентный спуск](#Стохастический-градиентный-спуск)
    2. [Адаптивный линейный элемент](#Адаптивный-линейный-элемент)
    3. [Персептрон Розенблатта](#Персептрон-Розенблатта)
    4. [Логистическая регрессия](#Логистическая-регрессия)
    5. [Сравнение алгоритмов](#Сравнение-алгоритмов)
    6. [Метод опорных векторов](#Метод-опорных-векторов)

## Метрические алгоритмы классификации 

[Оглавление](#Оглавление)

<table>
  <tr>
    <th>Метод</th>
    <th>Параметры</th>
    <th>Точность</th>
  </tr>
  <tr>
    <td>1nn</td>
    <td>k = 1</td>
    <td>0.0467</td>
  </tr>
  <tr>
    <td><strong>knn</strong></td>
    <td><strong>k = 6</strong></td>
    <td><strong>0.0333</strong></td>
  </tr>
  <tr>
    <td>kwnn</td>
    <td>k = 6, q = 0.56</td>
    <td>0.04</td>
  </tr>
  <tr>
    <td>PW (all kernels, except Gauss)</td>
    <td>h = 0.4</td>
    <td>0.04</td>
  </tr>
  <tr>
    <td>PW Gauss</td>
    <td>h = 0.1</td>
    <td>0.04</td>
  </tr>
  <tr>
    <td>MPF (all kernels, except Gauss)</td>
    <td>h = 1</td>
    <td>0.0467</td>
  </tr>
</table>

<p>
  Все метрические алгоритмы классификации будем рассматривать на датасете "Ирисы Фишера", а конкретнее - тренировочная выборка по ширине и длине лепестка и виду ириса. В выборке будет 150 цветков. Вот так, собственно, выглядит наш обучающий набор.
</p>

![set_of_irises](https://github.com/jelupre/ML1/blob/master/images/irises.png)

<p>
  kNN расшифровывается как k Nearest Neighbor или k Ближайших Соседей — это один из самых простых алгоритмов классификации, также иногда используемый в задачах       
  регрессии. Благодаря своей простоте, он является хорошим примером, с которого можно начать знакомство с областью Machine Learning. 
</p>

<p>
  Задача классификации в машинном обучении — это задача отнесения объекта к одному из заранее определенных классов на основании его формализованных признаков. 
  Каждый из объектов в этой задаче представляется в виде вектора в N-мерном пространстве, каждое измерение в котором представляет собой описание одного из признаков 
  объекта. 
</p>

[Оглавление](#Оглавление)

### Метод ближайшего соседа

[Оглавление](#Оглавление)

<p>
  Рассмотрим задачу kNN при k = 1 на языке R.
</p>

<h3>Алгоритм</h3>

Для классификации каждого из объектов тестовой выборки необходимо последовательно выполнить следующие операции:
<ul>
  <li>Вычислить расстояние до каждого из объектов обучающей выборки</li>
  <li>Найти объект из тренировочной выборки, от которого расстояние до классифицируемого объекта будет минимальным</li>
  <li>Класс классифицируемого объекта — это класс, ближайшего к нему объекта из обучающей выборки</li>
</ul>

<p>
  Введём число точек в тестовой выборке (n). Теперь создадим n точек с ограничениями по длине и ширине лепестка при помощи cbind и runif. Отбразим 
  тренировочную выборку. Рисуя тестовые точки, запускаем алгоритм 1NN для определения принадлежности одному из трёх существующих классов. 
  В самой функции 1NN ищем ближайшего по Евклидову расстоянию соседа для текущей точки и возвращаем вид ириса для неё же. 
  На рисунке ниже показан результат 10 случайно выбранных точек.
</p>

![1nn_10points](https://github.com/jelupre/ML1/blob/master/images/1nn_10points.png)

<p>
  Рассмотрим программную реализацию функции 1NN на языке программирования R.  
</p>


```R
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

Посмотрим на карту классификации для 1NN.

![1nn_map](https://github.com/jelupre/ML1/blob/master/images/1nn_map.png)

[Оглавление](#Оглавление)

### Метод k-ближайших соседей

[Оглавление](#Оглавление)

<h3>Алгоритм</h3>

<p>Для классификации каждого из объектов тестовой выборки необходимо последовательно выполнить следующие операции:</p>

<ul>
    <li>Вычислить расстояние до каждого из объектов обучающей выборки</li>
    <li>Отобрать k объектов обучающей выборки, расстояние до которых минимально</li>
    <li>Класс классифицируемого объекта — это класс, наиболее часто встречающийся среди k ближайших соседей</li>
</ul>

<p>
  Работа алгоритма k-ближайших соседей начинается с сортировки всех элементов обучающей выборки по расстоянию относительно тестовой точки, далее отправляем в саму функцию kNN обучающую выборку, k и точку, которую нужно классифицировать. Проходим циклом от 1 до k и смотрим, каких ирисов больше. Относим классифицируемую точку к тому классу, которого среди этих k элементов больше. Ниже демонстрируется код на языке R.
</p>

```R
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
  Прежде чем запустить алгоритм kNN для тестовой выборки необходимо выбрать k, для этого будем использовать LOO (leave-one-out CV). Как он работает? На вход мы получаем тренировочную выборку. Поочерёдно "вытаскиваем" из набора по одной точке и для каждой точки и выборки без неё запускаем kNN, изменяя k от 1 до длины "новой" обучающей выборки. Сравниваем полученный класс точки с классом этой же точки из изначальной тренировочной выборки, если классы не совпадают, то увеличиваем количество ошибок на данном k на 1. Пройдя все точки по всем k, ищем минимальное количество ошибок, и индекс этого элемента и будет лучшим k для этой выборки.
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

![LOO_6nn](https://github.com/jelupre/ML1/blob/master/images/LOO_6nn.png) 

![LOO_6nn_near](https://github.com/jelupre/ML1/blob/master/images/LOO_6nn_near.png)

Для данной выборки LOO возвращает k, равный 6. Теперь запустим 6NN для 10 случаной выбранных точек.

![6nn_10points](https://github.com/jelupre/ML1/blob/master/images/6nn_10points.png)

Посмотрим на карту классификации для 6NN.

![6nn_map](https://github.com/jelupre/ML1/blob/master/images/6nn_map.png)

[Оглавление](#Оглавление)

### Метод k-ближайших взвешенных соседей

[Оглавление](#Оглавление)

Метод kwNN отличается от kNN тем, что в нём добавляется вес к каждой точке в выборке, потом этот вес суммируется относительно классов, и, по итогу, вес какого класса будет больше, к тому классу и будет отнесена классифицируемая точка. Будем использовать весовую функцию q^i, где q = (0, 1), i = \[1, k\]. Возьмём k = 6 и для него найдём оптимальный q с точностью 0.01.

```R
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

![LOO_6wNN](https://github.com/jelupre/ML1/blob/master/images/LOO_6wnn.png)

![LOO_6wNN_near](https://github.com/jelupre/ML1/blob/master/images/LOO_6wnn_near.png)

Сам алгоритм kwNN выглядит следующим образом:

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

![6wnn_10points](https://github.com/jelupre/ML1/blob/master/images/6wnn_10points.png)

Теперь посмотрим на карту классификации.

![6wnn_map](https://github.com/jelupre/ML1/blob/master/images/6wnn_map.png)

[Оглавление](#Оглавление)

### Преимущества kwNN

[Оглавление](#Оглавление)

Недостаток kNN в том, что максимум может достигаться сразу на нескольких классах. В задачах с двумя классами этого можно избежать, если взять нечётное k. Более общая тактика, которая годится и для случая многих классов — ввести строго убывающую последовательность вещественных весов, задающих вклад i-го соседа в классификацию. Наглядно это видно на этом примере:

![knn_first_error](https://github.com/jelupre/ML1/blob/master/images/knn_first_error.png)

Мы выбрали k = 4, наш алгоритм берёт 2 красные и 2 синие ближайшие точки и возвращает тот класс, на котором максимум встретился раньше. Рассмотрим тот же случай методом ближайших взвешенных соседей, k = 4 и q = 0.5.

![kwnn_first_error](https://github.com/jelupre/ML1/blob/master/images/kwnn_first_error.png)

Второй значимый недостаток kNN перед kwNN можно заметить, когда среди k точек, большинство точек одного класса находятся дальше от классифицируемой, чем меньшинство точек. В таком случае kNN присвоит классифицуруемой точке класс большинства.

![knn_second_error](https://github.com/jelupre/ML1/blob/master/images/knn_second_error.png)

Однако kwNN учитывает близость точек отностельно классифицируемой, поэтому такой ошибки он не допускает.

![kwnn_second_error](https://github.com/jelupre/ML1/blob/master/images/kwnn_second_error.png)

[Оглавление](#Оглавление)

### Метод парзеновского окна

[Оглавление](#Оглавление)

Давайте пойдем дальше и посмотрим, как можно еще обобщить метод ближайших соседей, чтобы справиться вот с теми двумя недостатками, которые были отмечены. Мы вольны выбирать функцию весов соседей, как мы хотим. Давайте сделаем так, чтобы вес соседа убывал по мере возрастания расстояния до него. Введем два новых понятия: это ядро, это функция положительная, не возрастающая на отрезке \[0, 1\], и ширина окна. И вот если мы функцию веса зададим как такую конструкцию — ядро от расстояния поделить на ширину окна, то получим такую вот взвешенную функцию, которая придает меньшие веса тем соседям, которые находятся дальше. И этот метод называется методом окна Парзена. Рассмотрим типы ядер. 

Прямоугольное ядро

![Formula_Rect](https://github.com/jelupre/ML1/blob/master/images/Formula_Rect.png)

Треугольное ядро

![Formula_Triangle](https://github.com/jelupre/ML1/blob/master/images/Formula_Triangle.png)

Ядро Епанечникова

![Formula_Epanech](https://github.com/jelupre/ML1/blob/master/images/Formula_Epanech.png)

Квартическое ядро

![Formula_Quadratic](https://github.com/jelupre/ML1/blob/master/images/Formula_Quadratic.png)

Гауссовское ядро

![Formula_Gauss](https://github.com/jelupre/ML1/blob/master/images/Formula_Gauss.png)

где
![Formula_r](https://github.com/jelupre/ML1/blob/master/images/Formula_r.png)

Рассмотрим программную реализацию:

```R
PW <- function(set, point, h) {
  
  weights <- matrix(0, 1, 3)
  row <- dim(set)[1]
  class <- c("setosa", "versicolor", "virginica")
  
  for (i in 1:row) {
    ##if (distance_of_Euclid(set[i, 1:2], point) <= h) {
      tmp <- K(set[i, 1:2], point, h)
      if (set[i, 3] == "setosa")      weights[1] <- weights[1] + tmp  
      if (set[i, 3] == "versicolor")  weights[2] <- weights[2] + tmp
      if (set[i, 3] == "virginica")   weights[3] <- weights[3] + tmp
    ##}
  }
  
  if (weights[1] + weights[2] + weights[3] == 0) {
    return("white")
  }
  else {
    return(class[which.max(weights)])
  }

}
```

Функция ядра:

```R
K <- function(x, y, h) {
  
  r <- distance_of_Euclid(x, y)/h
  E <- 3 * (1 - r^2) / 4
  Q <- 15 * (1 - r^2) / 16
  G <- ((2 * pi) ^ (-1 / 2)) * exp(-(r ^ 2) / 2)
  P <- 1/2
  Tr <- 1 - abs(r)
  
  #возращаем гауссовское ядро 
  return(G)
}
```

Так же, как и в прошлых алгоритмах, нам необходимо определить оптимальный h. Воспользуемся LOO.

```R
LOO_h <- function(arr) {
  
  row <- dim(arr)[1]
  
  Q <- matrix(0, 20, 1)
  
  for (i in 1:row) {
    
    point <- arr[i, 1:2]
    new_arr <- arr
    new_arr <- new_arr[-i, ]
    
    for (h in 1:20) {
      
      class <- PW(new_arr, point, h/10)
      
      if (class != arr[i, 3]) {
        
        Q[h] <- Q[h] + 1
        
      }
      
    }
    
  }
  
  print(Q)
  
  return(which.min(Q)/10)
  
}
```

Теперь посмотрим на карты классификаций и график LOO(h) для всех пяти ядер.

![PW_R_map](https://github.com/jelupre/ML1/blob/master/images/PW_R_map.png)
![LOO_Rect](https://github.com/jelupre/ML1/blob/master/images/LOO_Rect.png)

![PW_T_map](https://github.com/jelupre/ML1/blob/master/images/PW_T_map.png)
![LOO_Triangle](https://github.com/jelupre/ML1/blob/master/images/LOO_Triangle.png)

![PW_E_map](https://github.com/jelupre/ML1/blob/master/images/PW_E_map.png)
![LOO_Epanech](https://github.com/jelupre/ML1/blob/master/images/LOO_Epanech.png)

![PW_Q_map](https://github.com/jelupre/ML1/blob/master/images/PW_Q_map.png)
![LOO_Quadratic](https://github.com/jelupre/ML1/blob/master/images/LOO_Quadratic.png)

![PW_G_map](https://github.com/jelupre/ML1/blob/master/images/PW_G_map.png)
![LOO_Gauss](https://github.com/jelupre/ML1/blob/master/images/LOO_Gauss.png)

[Оглавление](#Оглавление)

### Метод потенциальных функций

[Оглавление](#Оглавление)

Если в методе парзеновского окна центр окна поместить в классифицируемый
объект, то получим метод потенциальных функций.

Теперь ширина окна зависит не от классифицируемого объекта, а от обучающего.

Данный алгоритм имеет достаточно богатый набор из 2l параметров. Зафиксируем h = 0.4 и будем использовать те же ядра, что и в методе парзеновского окна, кроме гауссовского.

Рассмотрим функцию, которая находит потенциалы.

```R
find_potentials <- function(set, h, eps) {
  
  row <- dim(set)[1]
  err <- row
  potentials <- matrix(0, row, 1)
  
  while (err > eps) {
    
    err <- 0
    j <- 1
    
    while (TRUE) {
      
      class <- PF(set, set[j, 1:2], h, potentials)
      
      if (set[j, 3] != class) {
        
        ##print(j)
        #print(class)
        #print(set[j, 3])
        
        potentials[j] <- potentials[j] + 1
        break
        
      }
      
      j <- j %% row + 1
      
    }
    
    for (i in 1:row) {
      
      class <- PF(set[-i, ], set[i, 1:2], h, potentials)
      
      if (set[i, 3] != class) {
        
        err <- err + 1
        
      }
      
    }
    
    print(err)
    #print(potentials)
    
    
  }
  
  return(potentials)
  
}
```

Данная функция и есть основа метода потенциальных функций, она возвращает класс точки отностительно обучающей выборки.

```R
PF <- function(set, point, h, potentials){
  
  row <- dim(set)[1]
  w <- matrix(0, 3, 1)
  names(w) = c("setosa", "versicolor", "virginica")
  
  for (i in 1:row) {
    
    if (distance_of_Euclid(set[i, 1:2], point) <= h) {
      
      tmp <- K(set[i, 1:2], point, h)
      
      w[set[i, 3]] <- w[set[i, 3]] + potentials[i] * tmp
      
    }
    
  }
  
  if (max(w) == 0) {
    
    return("NA")
    
  }
  else {
    
    return(names(which.max(w)))
    
  }
  
}
```

Теперь взглянем на результаты работы. Будем отображать карту потенциалов и карту классификации для каждого ядра.

![PFC_R_map.png](https://github.com/jelupre/ML1/blob/master/images/PFC_R_map.png)
![PFC_map_potential_Rect.png](https://github.com/jelupre/ML1/blob/master/images/PFC_map_potential_Rect.png)

![PFC_T_map.png](https://github.com/jelupre/ML1/blob/master/images/PFC_T_map.png)
![PFC_map_potential_Triangle.png](https://github.com/jelupre/ML1/blob/master/images/PFC_map_potential_Triangle.png)

![PFC_E_map.png](https://github.com/jelupre/ML1/blob/master/images/PFC_E_map.png)
![PFC_map_potential_Epanch.png](https://github.com/jelupre/ML1/blob/master/images/PFC_map_potential_Epanech.png)

![PFC_Q_map.png](https://github.com/jelupre/ML1/blob/master/images/PFC_Q_map.png)
![PFC_map_potential_Quartic.png](https://github.com/jelupre/ML1/blob/master/images/PFC_map_potential_Quartic.png)

[Оглавление](#Оглавление)

## Байесовские методы классификации

[Оглавление](#Оглавление)

<table>
  <tr>
    <th>Метод</th>
    <th>Параметры</th>
    <th>Точность</th>
  </tr>
  <tr>
    <td>ННБК</td>
    <td></td>
    <td>0.04</td>
  </tr>
</table>

[Оглавление](#Оглавление)

### Линии уровня нормального распределения

[Оглавление](#Оглавление)

Рассмотрим многомерное нормальное распределение.
Вероятностное распределение с плотностью

![Formula_of_Gauss_distribution.png](https://github.com/jelupre/ML1/blob/master/images/Formula_of_Gauss_distribution.png)

называется <b>n-мерным многомерным нормальным (гауссовским) распределением</b> с математическим ожиданием (центром) µ (вектор из n вещественных чисел)  и ковариационной вещественной матрицей Σ размерностью n на n. Предполагается, что матрица Σ симметричная,
невырожденная, положительно определенная.


1) Рассмотрим случай, когда признаки некоррелированы, тогда линии уровня плотности распределения имеют форму эллипсоидов с центром µ и осями, параллельными осям координат.
Пусть 

![S1.png](https://github.com/jelupre/ML1/blob/master/images/S1.png)

![Sigma1.png](https://github.com/jelupre/ML1/blob/master/images/Sigma1.png)

2) Если признаки имеют одинаковые дисперсии, тогда эллипсоиды являются сферами.

![S2.png](https://github.com/jelupre/ML1/blob/master/images/S2.png)

![Sigma2.png](https://github.com/jelupre/ML1/blob/master/images/Sigma2.png)

3) Если признаки коррелированы, то матрица Σ не диагональна и линии уровня имеют форму эллипсоидов, оси которых повернуты (направлены вдоль собственных векторов матрицы Σ) относительно исходной системы координат.

![S3.png](https://github.com/jelupre/ML1/blob/master/images/S3.png)

![Sigma3.png](https://github.com/jelupre/ML1/blob/master/images/Sigma3.png)

Рассмотрим общую формулу:

![Eq.png](https://github.com/jelupre/ML1/blob/master/images/Eq.png)

[Оглавление](#Оглавление)

### Наивный нормальный байесовский классификатор

[Оглавление](#Оглавление)

Будем полагать, что все объекты описываются <i>n</i> числовыми признаками, то есть 

![object_x](https://github.com/jelupre/ML1/blob/master/images/object_x.jpg)

Предположим, что все признаки являются независимыми случайными величинами. Следовательно, функции правдоподбия классов представимы в виде

![multi](https://github.com/jelupre/ML1/blob/master/images/multi.jpg)

где каждый множитель - плотность распределений значений <i>j</i>-го признака для класса <i>y</i>, а полученное произведение - эмпирическая оценка одномерной плотности.

Оценивать <i>n</i> одномерных плотностей гораздо проще, чем одну <i>n</i>-мерную
плотность. Однако данное предположение крайне редко работает на практике,
поэтому такие алгоритмы называют <b>наивными байесовскими</b>.

Прежде, чем вывести формулу наивного байесовского классификатора, вспомним как выглядит формула оптимального байесовского классификатора:

![optimal_Bayes](https://github.com/jelupre/ML1/blob/master/images/optimal_Bayes.jpg)

Подставим эмпирические оценки одномерных плотностей в оптимальный байесовский классификатор и прологарифмируем для простоты вычислений. Получим наивный байесовский классификатор.

![naive_Bayes](https://github.com/jelupre/ML1/blob/master/images/naive_Bayes.jpg)

<b>Наивный нормальный байесовский классификатор.</b>  Предположим, что все признаки независимы и нормально распределены:

![Gauss_dis](https://github.com/jelupre/ML1/blob/master/images/Gauss_dis.jpg)

Рассмотрим программную реализацию:

```R
naive_Bayes <- function(Py, lambda, n, m, mu, sigma, point){
  
  p <- rep(0, m)
  
  for (i in 1:m) {
    
    p[i] <- Py[i] * lambda[i]
    
    for (j in 1:n) {
      
      p[i] <- p[i] * exp((-(point[j] - mu[i, j])^2 * (1/sigma[i, j]))/2) / sqrt(2 * pi * sigma[i, j])
      
    }
    
  }
  
  return(classes[which.max(p)])
  
}

# инициализация

set <- iris[ , 3:5]

row <- dim(set)[1]
col <- dim(set)[2]

num_classes <- table(set[3])
classes <- unique(set[, 3])
colors <- c("setosa" = "red", "versicolor" = "green", "virginica" = "blue")

# количество признаков и количество классов
n <- col - 1 
m <- dim(num_classes)

# априорная вероятность
Py <- rep(0, m) 

for (i in 1:m) {
  Py[i] <- num_classes[i] / row
}

lambda <- c(1, 1, 1)

# матожидание и дисперсия
mu <- matrix(0, m, n)
sigma <- matrix(0, m, n)

for (i in 1:m) {
  
  for (j in 1:n) {
    
    #print(set[set[, 3] == classes[i], ][, j])
    
    mu[i, j] <- mean(set[set[, 3] == classes[i], ][ , j])
    sigma[i, j] <- var(set[set[, 3] == classes[i], ][ , j])
    
  }
  
}
```

Посмотрим на карту классификации данного классификатора на примере ирисов Фишера.

![naive_norm_Bayes_map](https://github.com/jelupre/ML1/blob/master/images/naive_norm_Bayes_map.png)

Попробуем варьировать "лямбды":

1) Увеличим лямбду для setosa

![naive_norm_Bayes_map_setosa](https://github.com/jelupre/ML1/blob/master/images/naive_norm_Bayes_map_setosa.png)

2) Теперь сделаем максимальной лямбду для versicolor

![naive_norm_Bayes_map_versicolor](https://github.com/jelupre/ML1/blob/master/images/naive_norm_Bayes_map_versicolor.png)

3) "Выровняем" лямбды для versicolor и virginica, а для setosa установим меньше, чем у других

![naive_norm_Bayes_map_versicolor_and_virginica](https://github.com/jelupre/ML1/blob/master/images/naive_norm_Bayes_map_versicolor_and_virginica.png)

4) И на последок, увеличим лямбду для virginica

![naive_norm_Bayes_map_virginica](https://github.com/jelupre/ML1/blob/master/images/naive_norm_Bayes_map_virginica.png)

[Оглавление](#Оглавление)

### Подстановочный алгоритм

[Оглавление](#Оглавление)

Оценим параметры функций правдоподобия по частям обучающей выборки для каждого класса.

![mu_and_sigma_for_plugin](https://github.com/jelupre/ML1/blob/master/images/mu_and_sigma_for_plugin.png)

Затем эти выборочные оценки подставим в оптимальный байесовский классификатор. Получим байесовский
нормальный классификатор, который называется также <i>подстановочным (plug-in)</i>.

При длине выборки, стремящейся к бесконечности, оценки обладают рядом оптимальных
свойств: они <b>не смещены, состоятельны и эффективны</b>. Однако оценки,
сделанные по коротким выборкам, могут быть не достаточно точными.

Рассмотрим программную реализацию для ирисов Фишера. Здесь и далее будем считать классы равнозначными.

```R
plugin <- function(Py, lambda, n, m, mu, sigma, point) {
  
  point <- as.numeric(point)
  p <- rep(0, m)
  for (i in 1:m) {
    
    p[i] <- Py[i] * lambda[i]
    S <- matrix(0, n, n)
    for (i1 in 1:n) {
      for (i2 in 1:n) {
        
        S[i1, i2] <- sigma[i1, i2 + (i - 1) * n]
        
      }
    }
    
    p[i] <- p[i] *  exp(-(1/2) * t(point - mu[i, ]) %*% solve(S) %*% (point - mu[i, ])) / sqrt((2 * pi)^n * det(S))
    
  }
  
  return(classes[which.max(p)])
}
```

Взглянем на карту классификации и на разделяющие кривые.

![plugin_map](https://github.com/jelupre/ML1/blob/master/images/plugin_map.png)

Подробнее остановимся на разделяющих кривых.

1) Эллипс

![separating_curves_ellipse](https://github.com/jelupre/ML1/blob/master/images/separating_curves_ellipse.png)

2) Парабола

![separating_curves_parabola](https://github.com/jelupre/ML1/blob/master/images/separating_curves_parabola.png)

3) Гипербола

![separating_curves_hyperbole](https://github.com/jelupre/ML1/blob/master/images/separating_curves_hyperbole.png)

[Оглавление](#Оглавление)

### Линейный дискриминант Фишера

[Оглавление](#Оглавление)

Фишер (1936 г.) предложил простую эвристику <b>«Ковариационные матрицы классов равны»</b>, позволяющую увеличить число объектов, по которым
оценивается ковариационная матрица, повысить её устойчивость и заодно упростить алгоритм обучения.

Пусть ковариационные матрицы классов одинаковы и равны, тогда 

![sigma_for_LDF](https://github.com/jelupre/ML1/blob/master/images/sigma_for_LDF.png)

В этом случае разделяющая поверхность линейна (кусочно-линейна). Подстановочный алгоритм имеет вид:

![separating_plane](https://github.com/jelupre/ML1/blob/master/images/separating_plane.png)

Этот алгоритм называется <i>линейным дискриминантом Фишера (ЛДФ)</i>.
Он неплохо работает, когда формы классов действительно близки к нормальным и не слишком сильно различаются. В этом случае линейное решающее
правило близко к оптимальному байесовскому, но существенно более устойчиво, чем квадратичное, и часто обладает лучшей обобщающей способностью.

Рассмотрим программную реализацию всё на тех же ирисах Фишера. Существенных отличий от plug-in алгоритма здесь нет, за исключением лишь вычисления ковариационной матрицы.

```R
LDF <- function(Py, lambda, n, m, mu, sigma, point) {
  
  point <- as.numeric(point)
  p <- rep(0, m)
  
  for (i in 1:m) {
    
    p[i] <- Py[i] * lambda[i]
    p[i] <- p[i] *  exp(-(1/2) * t(point - mu[i, ]) %*% solve(sigma) %*% (point - mu[i, ])) / sqrt((2 * pi)^n * det(sigma))
    
  }
  
  return(classes[which.max(p)])
}


# матожидание и дисперсия
mu <- matrix(0, m, n)
sigma <- matrix(0, n, n)

# вычисление матожидания 
for (i in 1:m) {
  for (j in 1:n) {
    mu[i, j] <- mean(set[set[, n + 1] == classes[i], ][ , j])
  }
}

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
```

Теперь посмотрим, как выглядит карта классификации и разделяющие линии классов.

![LDF_map](https://github.com/jelupre/ML1/blob/master/images/LDF_map.png)

<b>Вывод</b>. Подстановочный алгоритм классификации является квадратичным дискриминантом. Он имеет ряд недостатков, одним из самых существенных из которых является плохая обусловленность или вырожденность матрицы ковариаций при малом количестве обучающих элементов класса, вследствие чего при обращении данной матрицы может получиться сильно искаженный результат, и весь алгоритм классификации окажется неустойчивым, будет работать плохо (возможна также ситуация, при которой обратная матрица вообще не будет существовать). Линейный дискриминант Фишера решает данную проблему.

[Оглавление](#Оглавление)

## Линейные методы классификации

[Оглавление](#Оглавление)

Пусть 

![XY_for_LC](https://github.com/jelupre/ML1/blob/master/images/XY_for_LC.png)

Если дискриминантная функция определяется как скалярное произведение вектора признаков и вектора пераметров w размерностью n, то получается 
<i>линейный классификатор</i>:

![LC](https://github.com/jelupre/ML1/blob/master/images/LC.png)

Уравнение ![Crossprod](https://github.com/jelupre/ML1/blob/master/images/Crossprod.png) задает гиперплоскость, разделяющую классы в пространстве. Если вектор x находится по одну сторону гиперплоскости с её направляющим вектором w, то объект x относится к классу +1, иначе — к классу −1.

[Оглавление](#Оглавление)

### Стохастический градиентный спуск

[Оглавление](#Оглавление)

![Set](https://github.com/jelupre/ML1/blob/master/images/Set.png)

Требуется найти вектор параметров w, при котором достигается минимум аппроксимированного эмпирического риска:

![Q](https://github.com/jelupre/ML1/blob/master/images/Q.png)

Каждый прецедент вносит аддитивный вклад в изменение вектора w, но вектор w изменяется только после перебора всех объектов. Сходимость
итерационного процесса можно улучшить, если выбирать прецеденты по одному, для каждого делать градиентный шаг и сразу обновлять вектор весов:

![Change_w](https://github.com/jelupre/ML1/blob/master/images/Change_w.png)

В <i>методе стохастического градиента (stochastic gradient, SG)</i> прецеденты перебираются в случайном порядке. 
Если же объекты предъявлять в некотором фиксированном порядке, процесс может зациклиться или разойтись.

Инициализация весов может производиться различными способами. Стандартная рекомендация — взять небольшие случайные значения:

![Initial_w](https://github.com/jelupre/ML1/blob/master/images/Initial_w.png)

Критерием останова может быть стабилизация Q и/или w.

Функционал изменяется по формуле:

![Change_Q](https://github.com/jelupre/ML1/blob/master/images/Change_Q.png), где

![Eps](https://github.com/jelupre/ML1/blob/master/images/Eps.png)

Темп обучения вычисляется по формуле

![Eta](https://github.com/jelupre/ML1/blob/master/images/Eta.png)

<h3>Алгоритм</h3>

<ol>
  <li>Инициализируем веса</li>
  <li>Инициализируем текущую оценку функционала</li>
  <li>Повторяем, пока Q и/или w не стабилизируется
    <ol>
        <li>Выбираем случайный объект</li>
        <li>Вычислить ошибку алгоритма:</li>
        <li>Объявляем вектор весов w</li>
        <li>Оценить новое значение функционала</li>
    </ol>
  </li>
</ol>

[Оглавление](#Оглавление)

### Адаптивный линейный элемент

[Оглавление](#Оглавление)

Возьмём функцию потерь:

![Loss_function_for_ADALINE](https://github.com/jelupre/ML1/blob/master/images/Loss_function_for_ADALINE.png)

![Derivative_of_loss_function_for_ADALINE](https://github.com/jelupre/ML1/blob/master/images/Derivative_of_loss_function_for_ADALINE.png)

Получаем правило обновления весов на каждой итерации метода стохастического градиента.

![Change_w_for_ADALINE](https://github.com/jelupre/ML1/blob/master/images/Change_w_for_ADALINE.png)

```R
# функция потерь ADALINE
L <- function(M) {
  return((M - 1)^2)
}

# выбор случайного объекта
i <- round(runif(1, 1, 100))
xi <- as.numeric(Xl[i, 1:n])
yi <- as.numeric(Xl[i, n + 1])

# ошибка
eps <- L(crossprod(w, xi) * yi)

# шаг градиентного спуска
eta <- as.numeric(1 / (sqrt(abs(xi[1])^2 + abs(xi[2])^2 + abs(xi[3])^2)))
w[1] <- w[1] - crossprod(w, xi) * yi * eta
w[2] <- w[2] - crossprod(w, xi) * yi * eta
w[3] <- w[3] - crossprod(w, xi) * yi * eta

# пересчёт Q
Q <- (1 - lambda) * Q + lambda * eps
```

Линейно-разделимая выборка:

![ADALINE](https://github.com/jelupre/ML1/blob/master/images/ADALINE.png)

![ADALINE_Q](https://github.com/jelupre/ML1/blob/master/images/ADALINE_Q.png)

Линейно-неразделимая выборка:

![ADALINE_2](https://github.com/jelupre/ML1/blob/master/images/ADALINE_2.png)

![ADALINE_Q_2](https://github.com/jelupre/ML1/blob/master/images/ADALINE_Q_2.png)

[Оглавление](#Оглавление)

### Персептрон Розенблатта

[Оглавление](#Оглавление)

В 1957 году Розенблатт предложил эвристический алгоритм обучения нейрона, основанный на принципах нейрофизиологии. Экспериментально было
установлено, что при синхронном возбуждении двух связанных нервных клеток синаптическая связь между ними усиливается. Чем чаще синапс
угадывает правильный ответ, тем сильнее становится связь. Своеобразная тренировка связи приводит к постепенному запоминанию информации. 
Если же синапс начинает часто ошибаться или вообще перестаёт использоваться, связь ослабевает, информация начинается забываться. 
Таким образом, память реализуется в синапсах. В математической модели нейрона роль памяти играет вектор синаптических весов w.

Формализуем данное правило. Пока будем считать признаки бинарными:

![XY_for_Perceptron](https://github.com/jelupre/ML1/blob/master/images/XY_for_Perceptron.png)

Тогда при классификации объекта получим <i>Правило Хэбба</i>:

![Hebb's_rule](https://github.com/jelupre/ML1/blob/master/images/Hebb's_rule.png)

В качестве функции потерь будем использзовать кусочно-линейную функцию

![Loss_function_for_Perceptron](https://github.com/jelupre/ML1/blob/master/images/Loss_function_for_Perceptron.png)

А шаг градиентного спуска будет изменяться согласно правилу Хэбба.

```R
# функция потерь
L <- function(M) {
  return(max(-M, 0))
}

# шаг градиентного спуска
eta <- as.numeric(1 / (sqrt(abs(xi[1])^2 + abs(xi[2])^2 + abs(xi[3])^2)))
w[1] <- w[1] + xi[1] * yi * eta
w[2] <- w[2] + xi[2] * yi * eta
w[3] <- w[3] + xi[3] * yi * eta
```

Линейно-разделимая выборка:

![Perceptron](https://github.com/jelupre/ML1/blob/master/images/Perceptron.png)

![Perceptron_Q](https://github.com/jelupre/ML1/blob/master/images/Perceptron_Q.png)

Линейно-неразделимая выборка:

![Perceptron_2](https://github.com/jelupre/ML1/blob/master/images/Perceptron_2.png)

![Perceptron_Q_2](https://github.com/jelupre/ML1/blob/master/images/Perceptron_Q_2.png)

[Оглавление](#Оглавление)

### Логистическая регрессия

[Оглавление](#Оглавление)

Метод логистической регрессии основан на довольно сильных вероятностных
предположениях, которые имеют несколько интересных последствий:

<ol>
    <li>линейный классификатор оказывается оптимальным байесовским;</li>
    <li>однозначно определяется функция потерь;</li>
    <li>можно вычислять не только принадлежность объектов классам, но также получать и численные оценки вероятности их принадлежности.</li>
</ol>

Рассмотрим оптимальный байесовский классификатор для двух классов {−1; +1}:

![Optimal_bayes_for_logis_regress](https://github.com/jelupre/ML1/blob/master/images/Optimal_bayes_for_logis_regress.png)

1. Байесовский классификатор является линейным:

![Bayes_linear](https://github.com/jelupre/ML1/blob/master/images/Bayes_linear.png)

2. Апостериорная вероятность принадлежности произвольного объекта классу может быть вычислена по значению дискриминантной функции:

![Sigmoid](https://github.com/jelupre/ML1/blob/master/images/Sigmoid.png)

Теперь рассмотрим метод стохастического градиента для логистической регрессии. Функция потерь будет выглядеть следующим образом:

![Loss_function_for_logis_regress](https://github.com/jelupre/ML1/blob/master/images/Loss_function_for_logis_regress.png)

А изменения вектора весов будут выглядеть так:

![Change_w_for_logis_regress](https://github.com/jelupre/ML1/blob/master/images/Change_w_for_logis_regress.png)

```R
# функция потерь
L <- function(M) {
  return(log2(1 + exp(-M)))
}

# сигмоидная функция
SigmoidFunction <- function(x) {
  return(1/(1 + exp(-x)))
}

# шаг градиентного спуска
eta <- as.numeric(1 / (sqrt(abs(xi[1])^2 + abs(xi[2])^2 + abs(xi[3])^2)))
w[1] <- w[1] + xi[1] * yi * eta * SigmoidFunction(crossprod(w, xi)*(-yi))
w[2] <- w[2] + xi[2] * yi * eta * SigmoidFunction(crossprod(w, xi)*(-yi))
w[3] <- w[3] + xi[3] * yi * eta * SigmoidFunction(crossprod(w, xi)*(-yi))
```

Линейно-разделимая выборка:

![Logis_regression](https://github.com/jelupre/ML1/blob/master/images/Logis_regression.png)

![Logis_regression_Q](https://github.com/jelupre/ML1/blob/master/images/Logis_regression_Q.png)

Линейно-неразделимая выборка:

![Logis_regression_2](https://github.com/jelupre/ML1/blob/master/images/Logis_regression_2.png)

![Logis_regression_Q_2](https://github.com/jelupre/ML1/blob/master/images/Logis_regression_Q_2.png)

[Оглавление](#Оглавление)

### Сравнение алгоритмов

[Оглавление](#Оглавление)

1. Как правило, логистическая регрессия дает лучшие результаты по сравнению с линейным дискриминантом Фишера (поскольку она основана
на менее жестких гипотезах), а также по сравнению с дельта-правилом и правилом Хэбба (поскольку она использует "более правильную" функцию потерь).

2. Для логистической регрессии есть возможность оценивать апостериорные вероятности и риски.

3. Градиентный метод обучения логистической регрессии наследует все недостатки метода стохастического градиента. Практичная реализация
должна предусматривать стандартизацию данных, отсев выбросов, регуляризацию (сокращение весов), отбор признаков, и другие эвристики
для улучшения сходимости. Возможно применение метода второго порядка, но он требует обращения n × n-матриц на каждом шаге и
также не застрахован от плохой сходимости.

Посмотрим, как три метода показывают себя на одной и той же выборке.

Линейно-разделимая выборка:

![Compare](https://github.com/jelupre/ML1/blob/master/images/Compare.png)

Линейно-неразделимая выборка:

![Compare_2](https://github.com/jelupre/ML1/blob/master/images/Compare_2.png)

[Оглавление](#Оглавление)

### Метод опорных векторов

[Оглавление](#Оглавление)

В настоящее время <b> метод опорных векторов (SVM) </b> считается одним из
лучших методов классификации!

Метод SVM обладает несколькими замечательными свойствами:
1. обучение SVM сводится к задаче квадратичного программирования,
имеющей единственное решение, которое вычисляется достаточно
эффективно даже на выборках в сотни тысяч объектов;
2. решение обладает свойством разреженности: положение оптимальной
разделяющей гиперплоскости зависит лишь от небольшой доли обучающих объектов. 
Они и называются опорными векторами, остальные объекты фактически не задействуются;
3. с помощью функции ядра метод обобщается на случай нелинейных
разделяющих поверхностей. Вопрос о выборе ядра, оптимального для
данной прикладной задачи, до сих пор остается открытой теоретической проблемой.

Пусть выборка линейно разделима и существуют значения параметров w и w0, 
при которых функционал числа ошибок принимает нулевое значение. 
Но тогда разделяющая гиперплоскость не единственна. 
Будем выбирать ее таким образом, чтобы она отстояла максимально 
далеко от ближайших к ней точек обоих классов.

Множество точек ![Set_for_line](https://github.com/jelupre/ML1/blob/master/images/Set_for_line.png) описывает полосу, разделяющую
классы.

Границами полосы служат две параллельные гиперплоскости с вектором
нормали w. Разделяющая гиперплоскость проходит ровно по середине между ними. 
Объекты, ближайшие к разделяющей гиперплоскости, лежат на
границах полосы, и именно на них достигается минимум. В каждом из
классов имеется хотя бы один такой объект.

Найдем ширину разделяющей полосы. Пусть x− и x+ — два обучающих
объекта классов −1 и +1 соответственно, лежащие на границе полосы. Тогда
ширина полосы

![Width_of_line](https://github.com/jelupre/ML1/blob/master/images/Width_of_line.png)

Ширина полосы максимальна, когда норма вектора w минимальна.

Итак, в случае линейно разделимой выборки получаем задачу линейного программирования:

![Problem_of_linear_programing](https://github.com/jelupre/ML1/blob/master/images/Problem_of_linear_programing.png)

Обобщим постановку задачи на случай линейно неразделимой выборки:
разрешим алгоритму допускать ошибки на обучающих объектах, но так,
чтобы ошибок было как можно меньше. Для этого введем дополнительные переменные, 
характеризующие величину ошибки на объектах обучения. Тогда будем иметь задачу квадратичного программирования:

![Problem_of_quadratic_programing](https://github.com/jelupre/ML1/blob/master/images/Problem_of_quadratic_programing.png)

Положительная константа C является управляющим параметром метода и
позволяет находить компромисс между максимизацией ширины разделяющей полосы и минимизацией суммарной ошибки.

Алгоритм допускает ошибку на объекте тогда и только тогда, когда отступ отрицателен. Если отступ лежит на отрезке (-1; 1), то объект 
попадает внутрь разделяющей полосы. Если отступ больше 1, то объект классифицируется правильно, и находится на некотором удалении от разделяющей
полосы.

Согласно теореме Куна-Таккера наша задача эквивалентна двойственной
задаче поиска седловой точки функции Лагранжа:

![Karush_Kuhn_Tucker](https://github.com/jelupre/ML1/blob/master/images/Karush_Kuhn_Tucker.png)

Функция Лагранжа:

![Function_of_Lagrange](https://github.com/jelupre/ML1/blob/master/images/Function_of_Lagrange.png)

![lambda](https://github.com/jelupre/ML1/blob/master/images/lambda.png) — вектор переменных, двойственных к w,
![etha](https://github.com/jelupre/ML1/blob/master/images/etha.png) — вектор переменных, двойственных к ![xi](https://github.com/jelupre/ML1/blob/master/images/xi.png)

Если ![lambda_i](https://github.com/jelupre/ML1/blob/master/images/lambda_i.png), то объект обучающей выборки называется <b> опорным вектором (support vector) </b>.

Все объекты делятся на следующие три типа:

1. Объект классифицируется правильно и не влияет на решение w. Такие объекты будем называть <b> периферийными </b> или <b> неинформативными </b>.
2. Объект классифицируется правильно и лежит в точности на границе разделяющей полосы. Такие объекты будем называть <b> опорными граничными </b>.
3. Объект либо лежит внутри разделяющей полосы, но классифицируется правильно, либо попадает на границу классов, либо вообще относится к чужому классу. 
Во всех этих случаях объект будем называть <b> опорным нарушителем </b>.

Лагранжиан выражается только через двойственные переменные ![lambda_i](https://github.com/jelupre/ML1/blob/master/images/lambda_i.png).

![Problem_of_quadratic_programing_2](https://github.com/jelupre/ML1/blob/master/images/Problem_of_quadratic_programing_2.png)

Здесь минимизируется квадратичный функционал, имеющий неотрицательно определенную квадратичную форму, следовательно, выпуклый. Область,
определяемая ограничениями неравенствами и одним равенством, также выпуклая. Следовательно, данная двойственная задача имеет единственное решение.

![w0](https://github.com/jelupre/ML1/blob/master/images/w0.png)

Алгоритм классификации выглядит следующим образом:

![Classificator_SVM](https://github.com/jelupre/ML1/blob/master/images/Classificator_SVM.png)

Обратим внимание, что суммирование идет не по всей выборке, а только по
опорным векторам, для которых ![lambda_i](https://github.com/jelupre/ML1/blob/master/images/lambda_i.png). 
Классификатор не изменится, если все остальные объекты исключить из выборки. Это свойство называют
<b>разреженностью (sparsity)</b>; именно оно и отличает SVM от других линейных классификаторов.

![lambda_i](https://github.com/jelupre/ML1/blob/master/images/lambda_i.png) обладают не только граничные опорные объекты, но и
объекты–нарушители. Это говорит о <b>недостаточной робастности (устойчивости к шуму)</b> SVM. Нарушителями могут быть объекты, появившиеся в
результате ошибочных наблюдений; их надо было бы исключить из выборки, а не строить по ним решение.

<b>Кривая ошибок</b> или <b>ROC-кривая</b> – графичекая характеристика качества бинарного классификатора, зависимость доли верных 
положительных классификаций от доли ложных положительных классификаций при варьировании порога решающего правила. ROC-кривая наглядно 
представляет, каким будет качество классификации при различных w0 и фиксированном w.

Пусть задана выборка объектов с соответствующими им верными ответами. Тогда для классификатора a(x) можно определить две характеристики качества:

1. Доля ложных положительных классификаций (<b>False Positive Rate, FPR</b>):

![FPR](https://github.com/jelupre/ML1/blob/master/images/FPR.png)

2. Доля верных положительных классификаций (<b>True Positive Rate, TPR</b>):

![TPR](https://github.com/jelupre/ML1/blob/master/images/TPR.png)

ROC-кривая показывает зависимость TPR от FPR при варьировании порога w0. Она проходит из точки (0,0), соответствующей максимальному 
значению w0, в точку (1,1), соответствующую минимальному значению w0. ROC-кривая монотонно не убывает. Чем выше лежит кривая, тем лучше качество классификации.

Площадь под ROC-кривой <b>AUC (Area Under Curve)</b> является агрегированной характеристикой качества классификации, 
не зависящей от соотношения цен ошибок. Чем больше значение AUC, тем «лучше» модель классификации. Данный показатель 
часто используется для сравнительного анализа нескольких моделей классификации.

В алгоритме будем использовать полиномиальное ядро.
Посмотрим, что показывает SVM на линейной-разделимой выборке.

![SVM_Linear](https://github.com/jelupre/ML1/blob/master/images/SVM_Linear.png)

![ROC_Linear](https://github.com/jelupre/ML1/blob/master/images/ROC_Linear.png)

Обратимся к линейно-неразделимой выборке, будем варьировать "штраф за ошибку" (С).

![SVM_Nonlinear_1](https://github.com/jelupre/ML1/blob/master/images/SVM_Nonlinear_1.png)

![ROC_Nonlinear_1](https://github.com/jelupre/ML1/blob/master/images/ROC_Nonlinear_1.png)

![SVM_Nonlinear_10](https://github.com/jelupre/ML1/blob/master/images/SVM_Nonlinear_10.png)

![ROC_Nonlinear_10](https://github.com/jelupre/ML1/blob/master/images/ROC_Nonlinear_10.png)

![SVM_Nonlinear_100](https://github.com/jelupre/ML1/blob/master/images/SVM_Nonlinear_100.png)

![ROC_Nonlinear_100](https://github.com/jelupre/ML1/blob/master/images/ROC_Nonlinear_100.png)

[Оглавление](#Оглавление)


