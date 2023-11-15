## Project goals
  - Find the key drivers of property value for single family properties.

## Project description
  - I've tasked to come up with a new model for determining tax value using features created by me for single family properties. This use a dataframe from Zillow in 2017.
## My Plan
 __Initial hypotheses__ 
  - Split into different counties since location could be a big factor in home value
  - Finished sqft and amount of bedrooms will make a difference in price
  - Look into bathrooms as well
## Acquire:
  - Acquire the data from SQL
  - You will need to use the properties_2017, predictions_2017, and propertylandusetype tables.
## Explore:
  - What affect does amount of bedrooms have on home values?
  - Are the amount of bathrooms dependent of home value or independent?
  - How much does finished area affect home values?
  - Does having a garage change the price?
## Modeling:
  - Use drivers in explore to build predictive models of different types
  - Evaluate models on train and validate data
  - Select the best model based on accuracy
  - Evaluate the test data
## Data dictionary:
| Feature | Definition |
| --- | --- |
| Bedrooms | Number of Bedrooms |
| Bathrooms | Number of Bathrooms |
| Finished Area | Usable Square Footage of Home |
| Home Value | Price of Home |
| Year Built | Year house was built |
| Lot Area | Square Footage of the entire property lot |
| County | County location of home |
| Garage Fits | Number of Cars that Fit in Garage |
| Garage Area | Square Footage of the Garage |
| Transaction Date | Last time the house was sold in 2017 |

## How to Reproduce: 
   - Clone this repo
   - Acquire data from MySql (Should make a zillow.csv after)
   - Run Notebook
## Key findings:
   - Bedrooms, Bathrooms, and Finished_area were the best features to use. Garages did not make the cut due to too much missing data and too many assumptions.
   - Model 4, Polynomial Regression with a degree of 3 worked best.
## Takeaways and Conclusions:
   - After running scaled data through the model
   - Test data ran on Model 4
   - 39.7% accuracy overall
   - Still quite low and only within 338k of the correct home value price which is alot of error.
## Recommendations: 
   - Recommend splitting data into counties and adding more features to the data collected in order to potentially predict Home Values in the future.
   - Simply having just bedrooms, bathrooms, and finished area are not enough to predict home values.