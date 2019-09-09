## load packages
library(data.table) # used for reading and manipulation of data
library(dplyr)      # used for data manipulation and joining
library(ggplot2)    # used for ploting 
library(caret)      # used for modeling
library(corrplot)   # used for making correlation plot
library(xgboost)    # used for building XGBoost model
library(cowplot)    # used for combining multiple plots 

## read datasets
train = fread("Train.csv")
test = fread("Test.csv")
submission = fread("SampleSubmission.csv")

## train data column names
names(train)

## test data column names
names(test)

## structure of train data
str(train)

## structure of test data
str(test)

#-------------------------------------------------------------------------------------------------------------------------
#Add Item_Outlet_Sales to test data
test[,Item_Outlet_Sales := NA] 

#Combining train and test datasets
combi = rbind(train, test) 
#-------------------------------------------------------------------------------------------------------------------------
## EDA - Univariate

#Item_Outlet_Sales - Target Variable
ggplot(train) + geom_histogram(aes(train$Item_Outlet_Sales), binwidth = 100, fill = "darkgreen") +
  xlab("Item_Outlet_Sales")
#Right skewed variable and would need some data transformation to treat skewness.


#Independent Variables (Numeric Variables)
#1. Item_Weight
p1 = ggplot(combi) + geom_histogram(aes(Item_Weight), binwidth = 0.5, fill = "blue")

#2. Item_Visibility
p2 = ggplot(combi) + geom_histogram(aes(Item_Visibility), binwidth = 0.005, fill = "blue")

#3. Item_MRP
p3 = ggplot(combi) + geom_histogram(aes(Item_MRP), binwidth = 1, fill = "blue")

#plot_grid() from cowplot package
plot_grid(p1, p2, p3, nrow = 1) 

#As you can see,there is no clear pattern in Item_Weight and Item_MRP.However,Item_Visibility is right
#skewed and should be transformed to curb its skewness
#-------------------------------------------------------------------------------------------------------------------------
#Independent Variables (categorical variables)
#1. Item Fat Content
ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")

#"Low Fat,low fat and LF" are same so combine and "Regular and reg" are same so combine. 
combi$Item_Fat_Content[combi$Item_Fat_Content == "LF"] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == "low fat"] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == "reg"] = "Regular"

#"Low Fat and Regular" only
ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")
#Most of the Item_Fat has low fat only
#-------------------------------------------------------------------------------------------------------------------------
#2. Item_Type
p4 = ggplot(combi %>% group_by(Item_Type) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Type, Count), stat = "identity", fill = "coral1") +
  xlab("") +
  geom_label(aes(Item_Type, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Item_Type")

p4
#Fruit and Vegetables and Snack foods are the two predominant products sold
#-------------------------------------------------------------------------------------------------------------------------
#3. Outlet_Identifier
p5 = ggplot(combi %>% group_by(Outlet_Identifier) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Identifier, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(Outlet_Identifier, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
p5
#10 different outlets and most of them are same
#-------------------------------------------------------------------------------------------------------------------------
#4. Outlet_Size
p6 = ggplot(combi %>% group_by(Outlet_Size) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Size, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(Outlet_Size, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
p6

#Medium is high and In Outlet_size's plot 4016 Observations,Outlet_Size is missing or blank.

second_row = plot_grid(p5, p6, nrow = 1)

plot_grid(p4, second_row, ncol = 1)
#-------------------------------------------------------------------------------------------------------------------------
#5. Outlet_Establishment_Year
p7 = ggplot(combi %>% group_by(Outlet_Establishment_Year) %>% summarise(Count = n())) + 
  geom_bar(aes(factor(Outlet_Establishment_Year), Count), stat = "identity", fill = "coral1") +
  geom_label(aes(factor(Outlet_Establishment_Year), Count, label = Count), vjust = 0.5) +
  xlab("Outlet_Establishment_Year") +
  theme(axis.text.x = element_text(size = 8.5))

p7
#there are lesser number of observations in the data for the outlets established in the year 1998 as compared
#to the other years.
#-------------------------------------------------------------------------------------------------------------------------
#6. Outlet_Type
p8 = ggplot(combi %>% group_by(Outlet_Type) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Type, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(factor(Outlet_Type), Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(size = 8.5))

p8
#Supermarket Type1 seems to be the most popular category of Outlet_Type

# ploting both plots together
plot_grid(p7, p8, ncol = 2)
#-------------------------------------------------------------------------------------------------------------------------
## EDA - Bivariate(Independent Variables affect the Dependent Variables)
#since we don't have target variable in test dataset we will restrict to training dataset
train = combi[1:nrow(train)]

#Here we will make use of scatter plots for continous or numeric variables and violin plots for
#the categorical variables

#Numeric variable-Numeric Variable(Target Variable)
# Item_Weight vs Item_Outlet_Sales
p9 = ggplot(train) + geom_point(aes(Item_Weight, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
     theme(axis.title = element_text(size = 8.5))

p9
#Item_Outlet_Sales is spread well across the entire range of the Item_Weight without any obvious pattern

# Item_Visibility vs Item_Outlet_Sales
p10 = ggplot(train) + geom_point(aes(Item_Visibility, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
      theme(axis.title = element_text(size = 8.5))

p10
#In the Item_Visibility vs Item_Outlet_Sales, there is a string of points at Item_Visibility = 0.0
#Which seems strange as item visibility cannot be completely zero.

# Item_MRP vs Item_Outlet_Sales
p11 = ggplot(train) + geom_point(aes(Item_MRP, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
      theme(axis.title = element_text(size = 8.5))

p11
#In Item_MRP vs Item_Outlet_Sales plot, we can clearly see 4 segments of prices that can be used
#in feature engineering to create a new variable.

second_row_2 = plot_grid(p10, p11, ncol = 2)

plot_grid(p9, second_row_2, nrow = 2)
#-------------------------------------------------------------------------------------------------------------------------
#Categorical Variables-Numeric Variables(Target Variable)

#check the distribution of the target variable across all the categories of each of the categorical variables

#Violin plots used instead of boxplots as they show the full distribution of the data.The horizontal
#width of a violin plot at a particular level indicates the concentration of data at that level.

#1. Item_Type vs Item_Outlet_Sales
#Box Plot
p12 = ggplot(train) + geom_boxplot(aes(Item_Type, Item_Outlet_Sales), fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 6),
        axis.title = element_text(size = 8.5))
p12

#Violin Plot
p12 = ggplot(train) + geom_violin(aes(Item_Type, Item_Outlet_Sales), fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 6),
        axis.title = element_text(size = 8.5))
p12

#1. Item_Fat_Content vs Item_Outlet_Sales
p13 = ggplot(train) + geom_violin(aes(Item_Fat_Content, Item_Outlet_Sales), fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))
p13

#3. Outlet_Identifier vs Item_Outlet_Sales
p14 = ggplot(train) + geom_violin(aes(Outlet_Identifier, Item_Outlet_Sales), fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))
p14

#Distribution of Item_Outlet_Sales across the categories of Item_Type is not very distinct and same
#is the case with Item_Fat_Content.
#However,the distribution of Outlet_Identifier from the rest of the categories of Outlet_Identifier.

second_row_3 = plot_grid(p13, p14, ncol = 2)

plot_grid(p12, second_row_3, ncol = 1)
#-------------------------------------------------------------------------------------------------------------------------
#Outlet_Size
#Distribution of target variable across the Outlet_Size
ggplot(train) + geom_violin(aes(Outlet_Size, Item_Outlet_Sales), fill = "magenta")

#the distribution of "Small" Outlet_Size is almost identical to the distribution of the blank category(First Violin)
#of the Outl_Size. So, we can substitute the blanks in Outlet_Size with "small".We will impute the values with "Small"

#Outlet_Location_Type
p15 = ggplot(train) + geom_violin(aes(Outlet_Location_Type, Item_Outlet_Sales), fill = "magenta")
p15
#Tier 1 and Tier 3 locations of Outlet_Location_Type look similar.

#Outlet_Type
p16 = ggplot(train) + geom_violin(aes(Outlet_Type, Item_Outlet_Sales), fill = "magenta")
p16
#In Outlet_Type, Grocery Store has most of its data points around the lower sales values as compared to
#other categories.Grocery will be keeping very low MRP Items.

plot_grid(p15, p16, ncol = 1)

#-------------------------------------------------------------------------------------------------------------------------
#Missing Value Treatment

#Checking for missing values in the columns of combi
colSums(is.na(combi))

#As you can see above, we have missing values in Item_Weight and Item_Outlet_Sales.
#Missing data in Item_Outlet_Sales can be ignored since they belong to the test dataset

#Imputing the missing values in Item_Weight column with mean weight based on the Item_Identifier variable.
missing_index = which(is.na(combi$Item_Weight))
for(i in missing_index){
  
  item = combi$Item_Identifier[i]
  combi$Item_Weight[i] = mean(combi$Item_Weight[combi$Item_Identifier == item], na.rm = T)
  
}

#Replacing 0 in Item_Visibility with the mean
zero_index = which(combi$Item_Visibility == 0)
for(i in zero_index){
  
  item = combi$Item_Identifier[i]
  combi$Item_Visibility[i] = mean(combi$Item_Visibility[combi$Item_Identifier == item], na.rm = T)
  
}
#-------------------------------------------------------------------------------------------------------------------------
## Feature Engineering
#FEATURE 1 - Item Type New
#FEATURE 2 - Item Type Category
#FEATURE 3 - Outlet Years
#FEATURE 4 - Price per unit Weight
#FEATURE 5 - Item MRP Clusters

#FEATURE 1 - Item Type New

#We can have a look at the Item_Type variable and classify the categories into perishable and non_perishable
#as per our understanding and make it into a new feature.

perishable = c("Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood")
non_perishable = c("Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene",
                   "Household", "Soft Drinks")

combi[,Item_Type_new := ifelse(Item_Type %in% perishable, "perishable",
                               ifelse(Item_Type %in% non_perishable, "non_perishable", "not_sure"))]

#FEATURE 2 - Item Type Category

#Compare them Item_Type with the first 2 characters of Item_Identifier, ie., 'DR','FD', and 'NC'.
#These identifiers most probably stand drinks,food and non-consumable.
#extracting first 2 characters for the first position
combi[,Item_category := substr(combi$Item_Identifier, 1, 2)]
combi$Item_Fat_Content[combi$Item_category == "NC"] = "Non-Edible"

#FEATURE 3 - Outlet Years

#We will also create a couple of more features - Outlet_Years(years of operation) and price_per_unit_wt(Price per unit weight)

#Years of operation of outlets
#Converting the Outlet_Establishement_Year to Outlet_Years by subtracting from 2013 year
#Outlet which is old will be popular so implementing Outlet_Years new Column
combi[,Outlet_Years := 2013 - Outlet_Establishment_Year]
combi$Outlet_Establishment_Year = as.factor(combi$Outlet_Establishment_Year)

#FEATURE 4 - Price per unit Weight
#Soft drinks for a particular 1.5 ltr has some discount nd if you buy in large quanitity and sales increases
combi[,price_per_unit_wt := Item_MRP/Item_Weight]

#FEATURE 5 - Item MRP Clusters
ggplot(train) + geom_point(aes(Item_MRP, Item_Outlet_Sales), colour = "violet", alpha = 0.3)
#Earlier in the Item_MRP vs Item_Outlet_Sales plot, we saw Item_MRP was spread across in 4 chunks.
#We can use K means clustering to create 4 groups using Item_MRP variable.
#We will go ahead with K=4.

Item_MRP_clusters = kmeans(combi$Item_MRP, centers = 4)
table(Item_MRP_clusters$cluster) # display no. of observations in each cluster

combi$Item_MRP_clusters = as.factor(Item_MRP_clusters$cluster)

#or group them manually
# combi[,Item_MRP_clusters := ifelse(Item_MRP < 69, "1st", 
#                                    ifelse(Item_MRP >= 69 & Item_MRP < 136, "2nd",
#                                           ifelse(Item_MRP >= 136 & Item_MRP < 203, "3rd", "4th")))]
#-------------------------------------------------------------------------------------------------------------------------
#Encoding Categorical Variables
#Label Encoding
#One Hot Encoding

#Label Encoding - Convert each category in a variable to a number
#               - More Suitable for ordinal variables - categorical variables with some order.
combi[,Outlet_Size_num := ifelse(Outlet_Size == "Small", 0,
                                 ifelse(Outlet_Size == "Medium", 1, 2))]

combi[,Outlet_Location_Type_num := ifelse(Outlet_Location_Type == "Tier 3", 0,
                                          ifelse(Outlet_Location_Type == "Tier 2", 1, 2))]

#removing categorical variables after label encoding
combi[, c("Outlet_Size", "Outlet_Location_Type") := NULL]
#-------------------------------------------------------------------------------------------------------------------------
## One Hot Encoding
#Each category of a categorical variable is converted into a binary column (1/0)
#Converted all the categorical columns with levels to numerical Columns with 1 and 0. example +> Columns = (n-1)levels
#Remove the following variable and then use dummyvars() function
ohe = dummyVars("~.", data = combi[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")], fullRank = T)
ohe_df = data.table(predict(ohe, combi[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")]))

combi = cbind(combi[,"Item_Identifier"], ohe_df)
#-------------------------------------------------------------------------------------------------------------------------
## Remove skewness
library(e1071) 
skewness(combi$Item_Visibility) 
skewness(combi$price_per_unit_wt)

combi[,Item_Visibility := log(Item_Visibility + 1)] # log + 1 to avoid division by zero
combi[,price_per_unit_wt := log(price_per_unit_wt + 1)]
#-------------------------------------------------------------------------------------------------------------------------
## Scaling and Centering data

#Which are the numerical variables - 29 numerical variables since we converted all using Encoding
num_vars = which(sapply(combi, is.numeric)) # index of numeric features
num_vars_names = names(num_vars)

#Remove Item_Outlet_Sales
combi_numeric = combi[,setdiff(num_vars_names, "Item_Outlet_Sales"), with = F]

#Preprocess function used for scaling
#Attribute which is different in units or we need to calculate distance in any algorithm
#Example Item_MRP since values are higher in that
prep_num = preProcess(combi_numeric, method=c("center", "scale"))
combi_numeric_norm = predict(prep_num, combi_numeric)
#-------------------------------------------------------------------------------------------------------------------------
#removing numeric independent variables
combi[,setdiff(num_vars_names, "Item_Outlet_Sales") := NULL] 
combi = cbind(combi, combi_numeric_norm)

#-------------------------------------------------------------------------------------------------------------------------
#Feature Selection
## splitting data back to train and test
train = combi[1:nrow(train)]
test = combi[(nrow(train) + 1):nrow(combi)]
test[,Item_Outlet_Sales := NULL] # removing Item_Outlet_Sales as it contains only NA for test dataset

## Correlation Plot
cor_train = cor(train[,-c("Item_Identifier")])
corrplot(cor_train, method = "pie", type = "lower", tl.cex = 0.9)

#Variables price_per_unit_wt and Item_Weight are highly correlated as the former one was created from
#the latter.Similarly price_per_unit_wt and Item_MRP are highly correlated for the same reason.

#Item_MRP,Item_MRP_Cluster__4, price_per_unit_wt and Outlet_IdentifierOUT019 has high correlation to Item_Outlet_Sales/
#This attributes are important for predicting Outlet_Sales.
#-------------------------------------------------------------------------------------------------------------------------
## Linear Regression

#All Independent Variables
linear_reg_mod = lm(Item_Outlet_Sales ~ ., data = train[,-c("Item_Identifier")])
summary(linear_reg_mod)

linear_reg_mod2 = lm(Item_Outlet_Sales ~ Item_MRP+Outlet_IdentifierOUT013+Outlet_IdentifierOUT017+Outlet_IdentifierOUT018+Outlet_IdentifierOUT027+Outlet_IdentifierOUT035+Outlet_IdentifierOUT045+Outlet_IdentifierOUT046+Outlet_IdentifierOUT049, data = train[,-c("Item_Identifier")])
summary(linear_reg_mod2)

## predicting on test set and writing a submission file
submission$Item_Outlet_Sales = predict(linear_reg_mod2, test[,-c("Item_Identifier")])

#-------------------------------------------------------------------------------------------------------------------------
## Lasso Regression
set.seed(1235)
my_control = trainControl(method="cv", number=5)
Grid = expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0002))

lasso_linear_reg_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales,
                       method='glmnet', trControl= my_control, tuneGrid = Grid)

# mean validation score
mean(lasso_linear_reg_mod$resample$RMSE)

#-------------------------------------------------------------------------------------------------------------------------
## Ridge Regression
set.seed(1236)
my_control = trainControl(method="cv", number=5)
Grid = expand.grid(alpha = 0, lambda = seq(0.001,0.1,by = 0.0002))

ridge_linear_reg_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales,
                       method='glmnet', trControl= my_control, tuneGrid = Grid)

# mean validation score
mean(ridge_linear_reg_mod$resample$RMSE)

#-------------------------------------------------------------------------------------------------------------------------
## RandomForest Model
set.seed(1237)
my_control = trainControl(method="cv", number=5)

tgrid = expand.grid(
  .mtry = c(3:10),
  .splitrule = "variance",
  .min.node.size = c(10,15,20)
)

#remove dependent and Item_Identifier
rf_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], 
               y = train$Item_Outlet_Sales,
               method='ranger', 
               trControl= my_control, 
               tuneGrid = tgrid,
               num.trees = 400,
               importance = "permutation")

# mean validation score
mean(rf_mod$resample$RMSE)

## plot displaying RMSE scores for different tuning parameters
plot(rf_mod)

## plot variable importance
plot(varImp(rf_mod))

#As expected Item_MRP is the most important variable in predicting the target variable.
#New features created by us price_per_unit_wt,Outlet_Years,Item_MRP_Clusters, are also among the top
#most important variables. this is why feature engineering plays a crucial role in predictive modelling.
#-------------------------------------------------------------------------------------------------------------------------
## List of parameters for XGBoost modeling
param_list = list(
        
        objective = "reg:linear",
        eta=0.01,
        gamma = 1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.5
        )

## converting train and test into xgb.DMatrix format
dtrain = xgb.DMatrix(data = as.matrix(train[,-c("Item_Identifier", "Item_Outlet_Sales")]), label= train$Item_Outlet_Sales)
dtest = xgb.DMatrix(data = as.matrix(test[,-c("Item_Identifier")]))

## 5-fold cross-validation to find optimal value of nrounds
set.seed(112)
xgbcv = xgb.cv(params = param_list, 
               data = dtrain, 
               nrounds = 1000, 
               nfold = 5, 
               print_every_n = 10, 
               early_stopping_rounds = 30, 
               maximize = F)

## training XGBoost model at nrounds = 428
xgb_model = xgb.train(data = dtrain, params = param_list, nrounds = 470)

## Variable Importance
var_imp = xgb.importance(feature_names = setdiff(names(train), c("Item_Identifier", "Item_Outlet_Sales")), 
                         model = xgb_model)

xgb.plot.importance(var_imp)
``
#Model Evaluation
Model <- c("Linear Regression","Lasso Regression","Ridge Regression","Random Forest","XGBoost")
RMSE_Score <- c(1128,1129,1135,1088,1090)
Model_Evaluation <- data.frame(Model,RMSE_Score)




