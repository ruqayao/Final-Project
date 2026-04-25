#Main Dataset
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class DurationCalorieAnalysis:
    """
    A class to analyze the relationship between workout duration and calories burned.
    """
    
    def __init__(self, file_path):
        """
        Initializes the analysis class with a file path.
        """
        self.file_path = file_path
        self.df = None
        self.correlation = None
        self.slope = None
        self.intercept = None
        self.r_squared = None

    def load_data(self):
        """
        Loads the data from the CSV file and cleans missing values 
        in the columns required for the analysis.
        """
        #Load the dataset using pandas
        self.df = pd.read_csv(self.file_path)
        
        #Target column names
        duration_col = 'Session_Duration (hours)'
        calories_col = 'Calories_Burned'
        
        #Drop rows where data is missing to ensure the math functions correctly
        self.df = self.df.dropna(subset=[duration_col, calories_col])
        
        #Ensure data is treated as numeric types
        self.df[duration_col] = pd.to_numeric(self.df[duration_col])
        self.df[calories_col] = pd.to_numeric(self.df[calories_col])

    def calculate_correlation(self):
        """
        Calculates the Pearson correlation coefficient between duration and calories.
        """
        x = self.df['Session_Duration (hours)']
        y = self.df['Calories_Burned']
        #Calculate correlation using pandas built-in method
        self.correlation = x.corr(y)
        return self.correlation

    def perform_regression(self):
        """
        Runs a simple linear regression to determine the predictive relationship.
        """
        x = self.df['Session_Duration (hours)']
        y = self.df['Calories_Burned']
        
        #Perform linear regression using scipy.stats
        result = stats.linregress(x, y)
        
        #Store results for summary and plotting
        self.slope = result.slope
        self.intercept = result.intercept
        self.r_squared = result.rvalue**2
        
        return result

    def create_visualization(self):
        """
        Creates a scatter plot with a regression line.
        """
        x = self.df['Session_Duration (hours)']
        y = self.df['Calories_Burned']
        
        #Calculate regression line values based on the formula y = mx + b
        regression_line = self.slope * x + self.intercept
        
        #Plotting the raw data points
        plt.scatter(x, y, alpha=0.5, color='blue', label='Individual Workouts')
        #Plotting the calculated regression line
        plt.plot(x, regression_line, color='red', linewidth=2, label='Trend Line')
        
        #Adding labels and titles for clarity
        plt.title('Impact of Workout Duration on Calories Burned')
        plt.xlabel('Duration (Hours)')
        plt.ylabel('Calories Burned')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
    def summarize(self):
        """
        Prints the results of the analysis.
        """
        print(" Research Question 3 Summary ")
        print("Correlation (r): {0:.4f}".format(self.correlation))
        print("R-squared (R2): {0:.4f}".format(self.r_squared))
        print("Equation: Calories = {0:.2f} * Duration + {1:.2f}".format(self.slope, self.intercept))
        print("Conclusion: Duration explains {0:.2f}% of calorie variance.".format(self.r_squared * 100))
        
        plt.show()

#Main execution block
if __name__ == "__main__":
    #Define the dataset path
    path = 'gym_members_exercise_tracking_synthetic_data.csv'
    
    #Instantiate the class
    analysis = DurationCalorieAnalysis(path)
    
    #Execute the analysis pipeline
    analysis.load_data()
    analysis.calculate_correlation()
    analysis.perform_regression()
    analysis.create_visualization()
    analysis.summarize()

#Reference Datasets
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class DurationAnalysis:
    """
    A class designed to analyze the relationship between workout 
    duration and the amount of calories burned.
    """
    
    def __init__(self, exercise_csv, calories_csv):
        #Constructor to initialize file paths and storage attributes.
        self.exercise_csv = exercise_csv
        self.calories_csv = calories_csv
        self.data = None
        self.correlation = None
        self.regression_results = None

    def load_and_prepare_data(self):
        #Method to load CSV files and merge them into a single dataset.
        try:
            #Loading the data using pandas
            exercise_df = pd.read_csv(self.exercise_csv)
            calories_df = pd.read_csv(self.calories_csv)
            
            #Merging the datasets based on User_ID
            self.data = pd.merge(exercise_df, calories_df, on='User_ID')
            print("Data successfully loaded and merged.")
        except Exception as e:
            print("An error occurred while loading data: " + str(e))

    def calculate_correlation(self):
        """
        Method to calculate the Pearson correlation coefficient.
        This measures the strength and direction of the linear relationship.
        """
        if self.data is not None:
            #Using the built-in corr() method from pandas
            self.correlation = self.data['Duration'].corr(self.data['Calories'])
            print("The Correlation Coefficient is: " + str(round(self.correlation, 4)))
            return self.correlation
        else:
            print("Error: Data must be loaded before calculating correlation.")
            return None

    def perform_linear_regression(self):
        """
        Method to run a simple linear regression.
        Predicts 'Calories' (Dependent variable) based on 'Duration' (Independent).
        """
        if self.data is not None:
            x = self.data['Duration']
            y = self.data['Calories']
            
            #Perform linear regression using scipy.stats
            #This returns the line slope, intercept, and the R-squared value
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            #Store results in a dictionary for easy access
            self.regression_results = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value
            }
            
            print("Linear Regression Analysis Complete.")
            print("Slope: " + str(round(slope, 4)))
            print("R-squared: " + str(round(self.regression_results['r_squared'], 4)))
            return self.regression_results
        else:
            print("Error: Data must be loaded before running regression.")
            return None

    def create_visualization(self):
        """
        Method to create a scatter plot with a regression line.
        """
        if self.data is not None and self.regression_results is not None:
            #Set plot size
            plt.figure(figsize=(10, 6))
            
            x = self.data['Duration']
            y = self.data['Calories']
            
            #Draw the scatter plot of the raw data
            plt.scatter(x, y, alpha=0.4, color='blue', label='Individual Workouts')
            
            #Calculate the regression line values (y = mx + b)
            m = self.regression_results['slope']
            b = self.regression_results['intercept']
            regression_line = m * x + b
            
            #Plot the regression line
            plt.plot(x, regression_line, color='red', linewidth=2, label='Regression Line')
            
            #Adding labels and title
            plt.title('Impact of Workout Duration on Calories Burned')
            plt.xlabel('Duration of Workout (Minutes)')
            plt.ylabel('Calories Burned')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()
            
#Execution Section 
#Instantiate the class with the file names
analysis = DurationAnalysis('raw_exercise.csv', 'raw_calories.csv')

#Execute the methods in order
analysis.load_and_prepare_data()
analysis.calculate_correlation()
analysis.perform_linear_regression()
analysis.create_visualization()





