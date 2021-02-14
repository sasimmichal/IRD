# Biblioteki

rm(list=ls()) # programowe czyszczenie œrodowiska

library(rpart) # do drzewa
library(rpart.plot) # do rysowania drzewa
#install.packages('ROCR')
library(ROCR) # do krzywej ROC
library(caret) # do waznosci zmiennych w modelu
library(dplyr)
library(tidyverse)

bank_fpath <- "C:/Users/Emilia Konopko/OneDrive/Pulpit/5 semestr/IRD LAB/data/bank_full.csv"
data <- read.csv2(bank_fpath, dec = ';')
data <- na.omit(data)

# Eksploracja danych
str(data)   #struktura danych
summary(data)

#y
require(dplyr)
data <- data %>%
mutate(y = ifelse(y == "no",0,1))
sum(data$y)

#data$y <- factor(ifelse(data$y == "yes",1,0)) zapisuje jako integer i nie mozna pozniej zrobic histogramu 

histogram
hist(data$y, main = "klient zapisal sie na lokate terminowa")

# U³amek liczby rekordów przeznaczony do zbioru testowego
test_prop <- 0.25
test_bound <- floor(nrow(data)* test_prop) #zaokraglamy w dol

data <- data[sample(nrow(data)), ]
data.test <- data[1:test_bound, ] #wybieramy wiersze od 1 do 11302 (zbior testowy)
data.train <- data[(test_bound+1):nrow(data), ] # od 11303 do konca (zbior uczacy)



# Eksploracja danych
names(data) #nazwy zmiennych 
str(data)
table(data$y) #sprawdzene ile osob wzielo lokate
prop.table(table(data$y))   # sprawdzenie odsetka

# Budowa drzewa decyzyjnego
tree1 <- rpart(y ~ age + marital +  housing +  duration +  pdays + previous,
              data=data.train,
              method="class",
              control = rpart.control(cp = 0.001, maxdepth = 4)) #g³êbia drzewa, korzeñ to depth=0

tree2 <- rpart(y ~ job + housing + contact + month + poutcome + duration,
               data=data.train,
               method="class",
               control = rpart.control(cp = 0.001, maxdepth = 4))

tree2 <- rpart(y ~ age + marital +  housing +  duration +  pdays + previous,
               data=data.train,
               method="class",
               control = rpart.control(cp = 0.001, maxdepth = 4))

# Wizualizacja drzewa
tree

plot(tree)
text(tree, pretty = TRUE)
rpart.plot(tree, under=FALSE, tweak=1.3, fallen.leaves = TRUE)

rpart.plot(tree2, under=FALSE, tweak=1.3, fallen.leaves = TRUE)
# IV - importance valeu
install.packages("InformationValue")
library(InformationValue)

IV(data$age, data$y, valueOfGood = 1)
IV(data$job, data$y, valueOfGood = 1)
IV(data$marital, data$y, valueOfGood = 1)
IV(data$education, data$y, valueOfGood = 1)
IV(data$default, data$y, valueOfGood = 1)
IV(data$balance, data$y, valueOfGood = 1)
IV(data$housing, data$y, valueOfGood = 1)
IV(data$contact, data$y, valueOfGood = 1)
IV(data$day, data$y, valueOfGood = 1)
IV(data$month, data$y, valueOfGood = 1)
IV(data$duration, data$y, valueOfGood = 1)
IV(data$campaign, data$y, valueOfGood = 1)
IV(data$pdays, data$y, valueOfGood = 1)
IV(data$previous, data$y, valueOfGood = 1)
IV(data$loan, data$y, valueOfGood = 1)
IV(data$poutcome, data$y, valueOfGood = 1)
# Interpretacja wyników

prop.table(table(data$housing, data$y), 1)
prop.table(table(data$housing, data$y), 2)


# Weryfikacja jakoœci klasyfikacji

EvaluateClassifier <- function(response_colname, prediction_colname, df,  positive = "1")
{
  y <- factor(df[response_colname][[1]]) # factor of positive / negative cases
  predictions <- factor(df[prediction_colname][[1]]) # factor of predictions
  precision <- posPredValue(predictions, y, positive)
  recall <- sensitivity(predictions, y, positive)
  F1 <- (2 * precision * recall) / (precision + recall)
  
  return(list(precision=precision, recall=recall, F1=F1))
}


# Weryfikacja jakoœci klasyfikacji - zbiór ucz¹cy i testowy

data.train$y_predicted <- predict(tree, data.train, type = "class") #predykcja na zbiorze uczacym
data.test$y_predicted <- predict(tree, data.test, type = "class")   # predykcja na zbiorze testowym

EvaluateClassifier('y', 'y_predicted', data.train)
EvaluateClassifier('y', 'y_predicted', data.test)

# Alternatywne drzewo decyzyjne - g³êbsze

tree_deeper <- rpart(y ~ age + job + marital + education + default + housing + loan + duration + campaign + pdays + previous,
                     data=data.train,
                     method="class",
                     control = list(maxdepth = 10))  #10 poziomow drzewa (ale R sam w sobie stwierdzil, ze rozwiazninie optymalne jest dla glebokosci 3)

rpart.plot(tree_deeper, under=FALSE, tweak=1.2, fallen.leaves = TRUE)

data.train$y_predicted_deeper <- predict(tree_deeper, data.train, type = "class")
data.test$y_predicted_deeper <- predict(tree_deeper, data.test, type = "class")

EvaluateClassifier('y', 'y_predicted_deeper', data.train)
EvaluateClassifier('y', 'y_predicted_deeper', data.test)

################################################################################
### Proba 2: Pakiet caret

library(caret)
set.seed(1)  #ziarno losowania

## podzial na zbior testowy i uczacy
inTraining <- createDataPartition(data$y, p = .8, list = FALSE) # p - % obserwacji
training <- data[ inTraining,]
#training  <- na.omit(training) # gdybysmy wczesniej nie usuneli z calego zbioru
testing  <- data[-inTraining,]

## budowa modelu na zbiorze uczacym - na poczatek okreslamy sposob uczenia

## 5-krotna walidacja krzyzowa
fitControl <- trainControl(    #sposob trenowania 
  method = "cv",              #walidacja krzyzowa
  number = 5)

# Najpierw proste drzewo z CV
#potrzebny pakiet e1071
#install.packages("e1071")
library(e1071)
treeCaret_simple <- train(y ~ age + job + marital + education + default + housing + loan + duration + campaign + pdays + previous,
                          data = training,  #jako zbior uczacy podajemy training 
                          method = "rpart", 
                          trControl = fitControl)

plot(treeCaret_simple)
rpart.plot(treeCaret_simple$finalModel)

confusionMatrix(data = as.factor(predict(treeCaret_simple, testing)), 
                reference = as.factor(testing$y), mode = "everything")


# Teraz recznie zadajemy zbior wartosci parametru zlozonosci do przeszukania
rpartGrid <- expand.grid(cp = seq(0.001, 0.1, by = 0.005))

treeCaret <- train(y ~ age + job + marital + education + default + housing + loan + duration + campaign + pdays + previous, 
                   data = training, 
                   method = "rpart", 
                   trControl = fitControl,
                   tuneGrid = rpartGrid)
treeCaret
# https://en.wikipedia.org/wiki/Cohen%27s_kappa
plot(treeCaret)
rpart.plot(treeCaret$finalModel)

# Ewaluacja
confusionMatrix(data = as.factor(predict(treeCaret, testing)), 
                reference = as.factor(testing$y), mode = "everything")

