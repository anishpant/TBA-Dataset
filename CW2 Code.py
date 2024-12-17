import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sma


df = pd.read_excel('CO2 Emissions due to cement India.xlsx')


#OLS regression and forecast for cement emissions 2020-2026
# Filter the data for the relevant years
filtered_df = df[df['Year'] <= 2020]
# Extract the independent and dependent variables
X = filtered_df['Year']
y = filtered_df['Cement emissions']

# Add a constant to the independent variable for OLS
X = sma.add_constant(X)
# Fit the OLS regression model
ols_model = sma.OLS(y, X).fit()
# Predict emissions for the years 2020 to 2026
future_years = pd.DataFrame({'Year': np.arange(2020, 2027)})
future_years_with_constant = sma.add_constant(future_years)
predictions = ols_model.predict(future_years_with_constant)

# Plot the actual data and the OLS predictions
plt.figure(figsize=(10, 6))

# Plot the actual emissions and production
plt.scatter(filtered_df['Year'], filtered_df['Cement emissions'], color='red', label='Actual Emissions', marker = 'o')

# Plot the OLS regression line for the historical data
predicted_values = ols_model.predict(X)
plt.plot(filtered_df['Year'], predicted_values, color='green', label='OLS Regression Line (2010-2019)', marker ='x' )

# Plot the OLS predictions for 2020 to 2026
plt.plot(future_years['Year'], predictions, color='green', linestyle='dashed', label='OLS Forecast (2020-2026)')

# Set the title and labels
plt.title('OLS Regression and Forecast for Cement Emissions (2020-2026)')
plt.xlabel('Year')
plt.ylabel('Metric Tons')
plt.legend() # Display the legend
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sma

#OLS regression and forecast for production volume of cement 2020-2026
# Filter the data for the relevant years
filtered_df = df[df['Year'] <= 2020]

# Extract the independent and dependent variables
X = filtered_df['Year']
y = filtered_df['Production volume of cement (in million metric tons)']

# Add a constant to the independent variable for OLS
X = sma.add_constant(X)
# Fit the OLS regression model
ols_model = sma.OLS(y, X).fit()
# Predict production volume for the years 2020 to 2026
future_years = pd.DataFrame({'Year': np.arange(2020, 2027)})
future_years_with_constant = sma.add_constant(future_years)
predictions = ols_model.predict(future_years_with_constant)
# Plot the actual data and the OLS predictions
plt.figure(figsize=(10, 6))

# Plot the actual production volume
plt.scatter(filtered_df['Year'], filtered_df['Production volume of cement (in million metric tons)'], 
            color='red', label='Actual Production', marker='o')

# Plot the OLS regression line for the historical data
predicted_values = ols_model.predict(X)
plt.plot(filtered_df['Year'], predicted_values, color='green', label='OLS Regression Line (2010-2019)', marker='x')

# Plot the OLS predictions for 2020 to 2026
plt.plot(future_years['Year'], predictions, color='green', linestyle='dashed', label='OLS Forecast (2020-2026)')

# Set the title and labels
plt.title('OLS Regression and Forecast for Cement Production (2020-2026)')
plt.xlabel('Year')
plt.ylabel('Production Volume (in million metric tons)')
plt.legend()
plt.show()



#Polynomial regression degree 2- cement emissions vs year
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
# Filter the dataset for years up to 2020
filtered_df = df[df['Year'] <= 2020]
# Extract independent (X) and dependent (y) variables
X = filtered_df[['Year']].values  # Ensure X is in 2D
y = filtered_df['Cement emissions'].values
# Define the range of polynomial degrees to visualize
degrees = [2]
# Plot the polynomial regression fits
plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='blue', label='Actual emissions')  # Actual data points
# Generate years from 2010 to 2026 for prediction
future_years = np.arange(2010, 2027).reshape(-1, 1)  # Years 2010 to 2026
# Loop through each degree and plot the corresponding polynomial regression
for degree in degrees:
    # Create and fit the polynomial regression model
    poly_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    poly_model.fit(X, y)
    
    # Generate predictions for both past and future years
    y_poly_pred = poly_model.predict(future_years)
    
    # Plot the polynomial regression line for both past and future years
    plt.plot(future_years, y_poly_pred, label=f'Degree {degree}')

# Set the title and labels
plt.title('Cement emissions vs Year (Polynomial Regression Forecast to 2026)')
plt.xlabel('Year')
plt.ylabel('Cement Emissions (Metric tons)')
plt.legend()
plt.show()



#Cement production vs year- polynomial regression degree 2
# Filter the dataset for years up to 2019
filtered_df = df[df['Year'] <= 2020]

# Extract independent (X) and dependent (y) variables
X = filtered_df[['Year']].values  # Ensure X is in 2D
y = filtered_df['Production volume of cement (in million metric tons)'].values

# Define the range of polynomial degrees to visualize
degrees = [2]

# Plot the polynomial regression fits
plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='blue', label='Actual Data')  # Actual data points

# Generate years from 2010 to 2026 for prediction
future_years = np.arange(2010, 2027).reshape(-1, 1)  # Years 2010 to 2026

# Loop through each degree and plot the corresponding polynomial regression
for degree in degrees:
    # Create and fit the polynomial regression model
    poly_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    poly_model.fit(X, y)
    
    # Generate predictions for both past and future years
    y_poly_pred = poly_model.predict(future_years)
    
    # Plot the polynomial regression line for both past and future years
    plt.plot(future_years, y_poly_pred, label=f'Degree {degree}')

# Set the title and labels
plt.title('Cement production vs Year (Polynomial Regression Forecast to 2026)')
plt.xlabel('Year')
plt.ylabel('Production volume of cement (in million metric tons)')
plt.legend()
plt.show()


#OLS regression (cement production vs cement emissions)
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Rename columns for easier handling
df_cleaned = df.rename(columns={
    "Cement emissions": "Cement_Emissions",
    "Production volume of cement (in million metric tons)": "Cement_Production"
})

# Select relevant columns
df_analysis = df_cleaned[["Cement_Emissions", "Cement_Production"]]

# Add a constant term for the OLS regression (intercept)
X = sm.add_constant(df_analysis["Cement_Production"])
Y = df_analysis["Cement_Emissions"]

# Fit the OLS model
ols_model = sm.OLS(Y, X).fit()

# Print the summary of the OLS regression
print(ols_model.summary())

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(df_analysis["Cement_Production"], df_analysis["Cement_Emissions"], color='blue', label='Actual Data')
plt.plot(df_analysis["Cement_Production"], ols_model.predict(X), color='red', label='OLS Fit')
plt.title('OLS Regression: Cement Production vs CO2 Emissions')
plt.xlabel('Cement Production (in million metric tons)')
plt.ylabel('Cement Emissions (metric tonnes)')
plt.legend()
plt.grid(True)
plt.show()




