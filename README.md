# xtern
xtern intern application - data science

Methods used: KNN clustering and linear regression.

1. Used KNN clustering model with restrictions such as rating >= 3.9, average cost <= $25 and cooking time <= 30 minutes to find all cuisines that fell into this range. With the KNN clustering I was able to find 8 centers that are around these restaurants and clustered the restaurants around their closest center.
2. Using a multiple linear regression model i found that the predictor variables Average cost, Minimum order, cook time, votes and reviews were significant to the response variable, which is rating. 
3. The Average cost and cook time had a positive relationship with the ratings; as the average cost and cook time increased, the rating increased. The minimum order did not affect the rating as heavily.
4. A linear regression model was used to predict the cook time of a restaurant based on the average price. As the average price went up, the cook time of the restaurant increased. 
