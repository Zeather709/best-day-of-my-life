library('tidyverse')

data(diamonds)
summary(diamonds)

price <- ggplot(data = diamonds, aes(x = price))
price + geom_histogram(binwidth = 250, colour = 'black', fill = 'mediumorchid') + labs(title = 'Cost of Diamonds in USD') 

cut <- ggplot(data = diamonds, aes(x = cut, y = price, fill = cut)) 
cut + geom_boxplot()

carat <- ggplot(data = diamonds, aes(x = carat, y = price, colour = color)) 
carat + geom_point(size = 0.5)

carat + stat_ecdf() 

clarity <- ggplot(data = diamonds, aes(x = clarity, y = price, fill = clarity)) 
clarity + geom_boxplot()
