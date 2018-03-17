
##STEP 1: Load Libraries
library(readxl)
library(stats)
library(ggplot2)
library(ggfortify)
library(zoo)
library(fpp2)
library(forecast)

##STEP 2: Load Data
# Read the data from Excel into R
exdata <- read_excel("data.xlsx")
mydata <- read_excel("exercise1.xlsx")
#data(oil)

#Part 1

##STEP 3: Explore Data - Preparation
# Look at the first few lines of mydata
head(mydata)

# Create a ts object called myts
myts <- ts(mydata[, 2:4], start = c(1981, 1), frequency = 4)
myts

#Create a ts object of just one series from the data
myts_GDP <- ts(mydata[, 4], start = c(1981, 1), frequency = 4)

##STEP 4: Explore Data - Visualization
#Plot the data with facetting, separates out each timeseries and plots together
autoplot(myts, facets = TRUE)

#Plot the data without facetting
autoplot(myts, facets = FALSE)

#Plot one series individually
autoplot(myts_GDP)

##STEP 5: Explore Data - Analysis
# Find the outliers' index position (max/min values in this case)
outlier_max <- which.max(myts)
outlier_max
outlier_min <- which.min(myts)
outlier_min

# Look at the seasonal frequencies of the series
#This was established when creating the TS object
frequency(myts)

#Create a season plot
#Seasonplots need a TS object that is univariate in nature
ggseasonplot(myts_GDP)

#Produce a polar coordinate season plot for the data
ggseasonplot(myts_GDP, polar = TRUE)

#Restrict the data to start in 1992
section <- window(myts_GDP, start = 1992)

#Autoplot the subset data
autoplot(section)

#Plot each quarter and average
ggsubseriesplot(section)

#Create a lag plot of the data
#A lag plot plots yt against yt-1 by default
#The correlations associated with lag plots form the autocorrelation function
gglagplot(myts_GDP)

#Create an ACF plot of the data
#See the correlations between observations
#Spikes outside blue lines might indicate exogenous factor influence on the data, inside can be ignored
ggAcf(myts_GDP)

#When data is seasonal or cyclic, the ACF will peak around the seasonal lags or at avg cycle length
#Keep in mind that seasonal periods are fixed, and cyclic periods can vary (but are longer than seasonal)

#Save the index position of the lag corresponding to maximum autocorrelation, the easy way
#Keep in mind that looks can be deceiving, and it's best not to eyeball this
#Create the Acf object
myts_GDP_ACF <- ggAcf(myts_GDP)
#Save only the ACF values
myts_GDP_ACF <- myts_GDP_ACF$data
#Find the max
myts_GDP_maxlag <- which.max(myts_GDP_ACF$ACF)
myts_GDP_maxlag_abs <- which.max(abs(myts_GDP_ACF$ACF))

#More on ACF
#Were you to generate a random value TS, it would be said to be white noise

#The blue lines are based on the sampling distribution for ACF assuming the data is white noise
#A series of ACF values that occur outside the blue lines indicate sufficient info for forecasting

#You can test all of the autocorrelations together
#Ljung-Box test looks at first h autocorrelation values together
#A p-value greater than 0.05 suggests data is not much different than white noise
#A p-value smaller than 0.05 suggests data is probably not white noise

#Plot the period to period (quarterly in this case) differences
autoplot(diff(myts_GDP))
#Now plot an ACF of those period differences
ggAcf(diff(myts_GDP))
#See all of those spikes outside of the blue lines at 4,8,12,16 and 2,6,10,14,18

#Now perform a Ljung-Box test on, say, 10 lag periods
Box.test(myts_GDP, lag = 10, type = "Ljung")
#See that p-value?  That's the p value of the ACF for the first 10 lags together (instead of going 1 by 1)
#But what about those quarterly changes?
Box.test(diff(myts_GDP), lag = 10, type = "Ljung")


# Part 2

#For forecasting, what actually happens is a series of possible futures is generated
#So, beyond the known, a bunch of different lines are drawn as a simulated future
#The mean (or median) of these simulated futures is the forecast

#Prediction intervals/forecast intervals are based on the percentiles of the simulated futures
#80% should contain 80% of the simulated futures.  20% should be outside the area
#Generally, half above the point forecast line and half below

#A naive forecast uses the most recent observation.
#While not good for production, it can be used to benchmarking

#Generate a naive forecast for the next 4 quarters
nfc_myts_GDP <- naive(myts_GDP, h=4)
nfc_myts_GDP
#Plot the naive forecast
autoplot(nfc_myts_GDP)

#For seasonal data, you can use the corresponding season from the last year
snfc_myts_GDP <- snaive(myts_GDP, h=4)
#Plot the seasonal naive forecast, and check a summary
autoplot(snfc_myts_GDP)
summary(snfc_myts_GDP)

#Fitted values are point forecasts using all data up to the previous point
#Think of them as moving, or step-by-step
#These are often not true forecasts because their parameters are estimated on all data

#A residual is the difference between an observed and forecasted value
#If the model is pretty good, the residuals should all look like white noise
#The residuals also should have a mean of zero, if they don't the forecast could be biased
#You can also assume that a good model's residuals will have constant variance and are normally distributed
#This is described as gaussian white noise

#Do a quick residuals check on the naive forecasts from earlier
checkresiduals(nfc_myts_GDP)
checkresiduals(snfc_myts_GDP)
#Notice that for the regular naive forecast, the residuals are normally distributed, but not white noise
#The residuals for the seasonal naive forecast are fairly normally distributed and mostly white noise

#Just like in normal ML, you use training and test sets
#Train on a large chunk of what's known, and test on a small holdout period
#Just because the forecast fits the training set well does not mean it's a good forecast
#Training sets can be created with subset() for index positions, or window() for time periods

#Forecast errors are the difference between forecasted values and observed values
#These are NOT residuals
#Forecast errors are in the test set, and residuals are in the training set
#Forecast errors can be for any forecast horizon residuals are based on one-step forecasts
#Metrics like MAE and MSE are good only for data of the same scale
#MAPE is only better if all values are non-zero and large enough.  MAPE also assumes a natural zero
#MASE is like MAE but is scaled and can be used across series

#Create the training data as train
train <- subset(myts_GDP, end = 96) #Hold out the last 4 quarters
#Compute naive forecasts and save to naive_fc
naive_fc <- naive(train, h = 4)
#Compute mean forecasts and save to mean_fc
mean_fc <- meanf(train, h = 4)
#Compute mean forecasts and save to mean_fc
snaive_fc <- snaive(train, h = 4)
#Use accuracy() to compute RMSE statistics
accuracy(naive_fc, myts_GDP)
accuracy(mean_fc, myts_GDP)
accuracy(snaive_fc, myts_GDP)
#In this example, the most accruacte is actually the meanf() forecast (based on RMSE)
#Normally you want a low RMSE and white noise residuals

#Create 3 different training sets using window()
#omit the last 1, 2, and 3 years
#train1 <- window(myts[, "GDP"], end = c(2004, 4))
train1 <- window(myts_GDP, end = c(2004, 4))
train2 <- window(myts_GDP, end = c(2003, 4))
train3 <- window(myts_GDP, end = c(2002, 4))

#Produce forecasts using snaive()
fc1 <- snaive(train1, h = 4)
fc2 <- snaive(train2, h = 4)
fc3 <- snaive(train3, h = 4)

#Use accuracy() to compare the test set MAPE of each series
#accuracy(fc1, myts[, "GDP"])["Test set", "MAPE"]
accuracy(fc1, myts_GDP)["Test set", "MAPE"]
accuracy(fc2, myts_GDP)["Test set", "MAPE"]
accuracy(fc3, myts_GDP)["Test set", "MAPE"]
#So, according to this, the snaive() model does better on forecasting 2003 than it does for the other two
#When forecasting 2004 and 2005 (fc2 and fc1 respectively) it seems to have a somewhat stable MAPE

#Time series cross validation
#A small test set test can be deceiving, and may not indicate a good forecast model
#"Forecasting accuracy on a rolling origin"
#In general, you use tsCV and choose the model with the smallest MSE

#Create a matrix to store the cross-validated forecasts for up to 4 periods out
e <- matrix(NA_real_, nrow = 1000, ncol = 4)
#Now do the cross-validation (snaive for now)
for (h in 1:4)
  e[, h] <- tsCV(myts_GDP, forecastfunction = snaive, h = h)

#Compute the MSE values and remove missing values
#Remember, for tsCV you have to define your metric function (MSE, MAPE, etc.)
mse <- colMeans(e^2, na.rm = TRUE)

#Plot the MSE values against the forecast horizon
data.frame(h = 1:4, MSE = mse) %>%
  ggplot(aes(x = h, y = MSE)) + geom_point()


# Part 3

#Exponentially weighted forecasts is commonly referred to as exponential smoothing
#Common notation is y(hat) t+h|t
#Forecasting h steps ahead given up to time t
#Alpha determines how much weight is placed on the most recent observation, and how quickly they decay
#Larger alpha = more weight on recent and they decay faster

#Use the ses (simple exponential smoothing) function to forecast the next 8 periods
fc <- ses(myts_GDP, h = 8)

#Use summary() to see the model parameters
summary(fc)

#Use autoplot() to plot the forecasts
autoplot(fc)

#Add the one-step forecasts for the training data to the plot
#autolayer adds a layer to the existing plot so both show at the same time
autoplot(fc) + autolayer(fitted(fc))
#Not super interesting for this example, but still shows what's going on

#Test out comparing ses to the naive from earlier
#Create a training set using subset
train <- subset(myts_GDP, end = length(myts_GDP) - 8)

#Compute SES and naive forecasts, save to fcses and fcnaive
fcses <- ses(train, h = 8)
fcnaive <- naive(train, h = 8)

#Calculate forecast accuracy measures
accuracy(fcses, myts_GDP)
accuracy(fcnaive, myts_GDP)
#So for these, the ses actually has a lower RMSE on the test set than the naive, neat!

#The ses function only accounts for forecast and level
#The holt's linear trend accounts for forecast, level, and trend
#It does, however, have a new smoothing parameter, beta
#If you add phi, a damping parameter, to this you get the damped trend method

#Test out a holt forecast
# Produce 4 period forecasts using holt()
fcholt <- holt(myts_GDP, h=4)

#Look at fitted model using summary()
summary(fcholt)

#Plot the forecasts
autoplot(fcholt)

#Check that the residuals look like white noise
checkresiduals(fcholt)
#Notice that the residuals still have many points in the ACF that are outside the blue lines

#The Holt Winters method is the same as above but adds a seasonality parameter
#The method also accounts for the seasonal component evolving over time
#This method contains the additive and multiplicative aspects

#Check out the holt winters method
#Plot the data
autoplot(myts_GDP)

#Produce 3 year forecasts
fc <- hw(myts_GDP, seasonal = "multiplicative", h = 4)

#Check if residuals look like white noise
checkresiduals(fc)

#Plot forecasts
autoplot(fc)

#Compare seasonal naive to the holt winters model
#Create training data with subset()
train <- subset(myts_GDP, end = 96)

# Holt-Winters additive forecasts as fchw
fchw <- hw(train, seasonal = "additive", h = 4)

# Seasonal naive forecasts as fcsn
fcsn <- snaive(train, h=4)

# Find better forecasts with accuracy()
accuracy(fchw, myts_GDP)
accuracy(fcsn, myts_GDP)

#Based on RMSE, it looks like the additive holt winters model wins!
#Look at a plot of the forecast and the fitted values it used
autoplot(fchw) + autolayer(fitted(fchw))

#Innovations state space models
#Another way of writing out exponential smoothing methods
#Each of Trend, Seasonal, and Error have different methods withing
#Trend has none,additive,additive-damped, seasonal has none,additive,multiplicative
#Error has additive,multiplicative
#These are referred to as ETS models (error, trend, seasonal)
#Within ets models, parameters are estimated using the likelihood or probability of data arising from the model
#For additive errors, this is equivalent to minimizing the SSE
#Normally you choose the best model by minimizing a corrected version of Akaike's Information Criterion (AIC)
#AICc is the bias corrected version.  It's faster than CVts

#Check out the ets method
#Fit ETS model
fitgdp <- ets(myts_GDP)

# Check residuals
checkresiduals(fitgdp)

# Plot forecasts
autoplot(forecast(fitgdp))

#Compare ets to the seasonal naive forecast with CVts
#Function to return ETS forecasts
fets <- function(y, h) {
  forecast(ets(y), h = h)
}

# Apply tsCV() for both methods
e1 <- tsCV(myts_GDP, forecastfunction=fets, h = 4)
e2 <- tsCV(myts_GDP, forecastfunction=snaive, h = 4)

# Compute MSE of resulting errors (watch out for missing values)
mean(e1^2, na.rm=TRUE)
mean(e2^2, na.rm=TRUE)
#According to the MSE, looks like ETS wins this round

#ETS doesn't work for everything though
#For some annual data with no seasonality or clear trend, the ets will produce a flat line
