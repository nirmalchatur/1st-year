import pandas as pd
import pandas as pd
import csv
import turtle
from datetime import date
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LinearRegression
print("################################################")
print("************************************************")
print("           BANK MANAGEMENT SYSTEM                ")
print("************************************************")
print("################################################")
pf=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")
transact= pf[['transaction','nationality','contact']]
def transactions():
     print(transact)
def withdraw():
                                        df=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")
                                        bb=df.loc[:,"balance"]
                                        aa=df.loc[:,"name"]
                                        concatenated = pd.concat([bb, aa], axis=1)
                                        print(concatenated)
                                        qq=int(input("which one:"))
                                        cc=df.at[qq,"balance"]
                                        aa=int(input("amount to be withraw:"))
                                        if aa>cc:
                                            print("Insufficient funds.")
                                        else:
                                            ss=cc-aa
                                            str(cc)
                                            str(ss)
                                            df["balance"].replace({cc:ss},inplace=True)
                                            print(df)
                                            print("Successfully withdrew ")
                                            
                                       

# Function to display transaction history
def display_transactions():
    print("Transaction History:")
    print(transact)
def minbal():
                                      df=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv",index_col=0)
                                      print('minimum balance')
                                      print()
                                      print(df.balance.min())
                                      print("****")
                                      print()
def maxbal():
                                      df=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")
                                      print(df)
                                      print("highest balance")
                                      print(df.balance.max())
                                      print("*********")
def bankcsv():
                                   print('reading file bank')
                                   print()
                                   print()
                                   df=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv",index_col=0)
                                   print(df)
                                   print("complete data is as shown")
                                   f1=open("/content/bank_mangement.csv",'r')
                                   listdata=[]    
                                   listdata=list(csv.reader(f1,delimiter=','))
                                   for i in listdata:
                                           print(i)
def removebank():
                                       print ('Deleting Account holder from file account')
                                       df=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv",index_col=0) 
                                       print(df)
                                       accno=int(input("enter accno"))
                                       df.drop(accno,axis=0,inplace=True)
                                       print(df)
     
def new_bankacc():
                                        
    df = pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv", index_col=0)
    name = input("Enter your name: ")
    age = input("Enter your age: ")
    job = input("Enter your job: ")
    marital = input("Enter your marital status: ")
    education = input("Enter your education level: ")
    default = input("Enter your default status: ")
    balance = input("Enter your balance: ")
    housing = input("Enter your housing status: ")
    contact = input("Enter your contact information: ")
    day = input("Enter the day: ")
    month = input("Enter the month: ")
    duration = input("Enter the duration: ")
    campaign = input("Enter the campaign: ")
    pdays = input("Enter the pdays: ")
    previous = input("Enter the previous: ")
    poutcome = input("Enter the poutcome: ")
    deposit = input("Enter the deposit: ")
    Region = input("Enter the Region: ")
    gender = input("Enter your gender: ")
    nationality = input("Enter your nationality: ")
    transaction = input("Enter the transaction: ")

    df1 = pd.DataFrame(
        {
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'balance': [balance],
            'housing': [housing],
            'contact': [contact],
            'day': [day],
            'month': [month],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
            'deposit': [deposit],
            'Region': [Region],
            'name': [name],
            'gender': [gender],
            'nationality': [nationality],
            'transaction': [transaction]
        }
    )
    print(df1)

    result = pd.concat([df, df1], axis=0)
    print(result)
                                       
def sort_names():
                                    print("sort by ascending ")
                                    df=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv",index_col=0)
                                    df=df.sort_values('balance')
                                    print(df)
    
def deposit():
                                     
                                     df = pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")
                                     bb = df.loc[:, "job"]
                                     aa = df.loc[:, "duration"]
                                     

                                     concatenated = pd.concat([bb, aa], axis=1)
                                     print(concatenated)

                                     
                                    
                        
                    
                                     qq=int(input("which one:"))
                                     cc=df.at[qq,"balance"]
                                     aa=int(input("amount to be deposited:"))
                                     ss=cc+aa
                                     str(cc)
                                     str(ss)
                                     df["balance"].replace({cc:ss},inplace=True)
                                     print(df)
def access():
                                     df=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")
                                     print(df)
                                     s=int(input("which one:"))
                                     a= df.iloc[s]
                                     print(a)


def get_balanceabove():   #filters the DataFrame based on the 'balance' column using the condition vv['balance'] > n. The resulting filtered DataFrame is stored in the variable group and printed to the console.
    print()
    vv = pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")
    n = int(input("Enter the amount above: "))
    group = vv[vv['balance'] > n]
    print(group)
    group = vv.loc[vv['balance'] > n]
    print(group)


def head():  
    print(pf.head(5))

  
def barplot():
      plt.bar(pf['duration'], pf['day'])

       # Customize the plot
      plt.xlabel('duration')
      plt.ylabel('day')
      plt.title('Bar Graph')

         # Display the plot
      plt.show()
      print()
def info():
    print()
def describe():
    print(pf.describe())
def group_marital():
    print(pf.groupby('marital'))
def group_month():
   print(pf.groupby('month'))
def namestart_with_s():
    mask = pf['Region'].str.startswith('S')
    count = mask.sum()
    print("Number of regions whose names start with 'S':", count)
def groupby_education():
    print()
    print("printing the values group according to the education data=\n")
    df4=pf.groupby('education')
    print(df4)
    print()
    print(df4.value_counts())
def groupby_marital():
    print()
    print("printing the values group according to the maritial data=\n")
    df1=pf.groupby('marital')
    print(df1)
    print()
    print(df1.value_counts())
def groupby_job():
    print()
    print("printing the values group according to the job data=\n")
    df2=pf.groupby('job')
    print(df2)
    print(df2.value_counts())
def groupby_poutcome():
    print()
    print("printing the values group according to the poutcome data=\n")
    df3=pf.groupby('poutcome')
    print(df3)
    print(df3.value_counts())
def duration_more():
    print()
    print("printing the data with most duration=\n")
    print()
    e=pf.loc[ (pf['duration']>80000) & (pf['duration'] <120000)]
    print(e.value_counts())
    print(type(e))
def deposite():
    print()
    counts = pf.loc[pf['deposite'] == 'yes', 'deposite'].value_counts()
    print(counts)
def line():
      df = pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\imdb data.csv")
      plt.plot(df['Certificate'],df['IMDB_Rating'])
      transact= pf[['age','day']]
# Set the title and labels for the graph
      plt.title('age vs day graph')
      plt.xlabel('Year')
      plt.ylabel('rating')
      
# Display the line graph
      plt.show()
      X = transact.values

# Specify the number of clusters
      k = 2

# Perform K-means clustering
      kmeans = KMeans(n_clusters=k, random_state=42)
      kmeans.fit(X)

# Get the cluster labels and cluster centers
      labels = kmeans.labels_
      centers = kmeans.cluster_centers_

# Plot the data points and cluster centers
      plt.scatter(X[:, 0], X[:, 1], c=labels)
      plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('age vs day graph using K means')
# Display the plot
      plt.show()
def findname():
     n = int(input("Enter your amount: "))
     filtered_df = pf[pf['balance'] == n]

     if not filtered_df.empty:
         print("It is present at index:", filtered_df.index)
     else:
         print("It is not present in the DataFrame.")
def pieplot():
# Plot the pie chart
         pf=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\imdb data.csv")
         plt.pie(pf['IMDB_Rating'],labels=pf['Certificate'], autopct='%1.1f%%')
         print("A-- fit for giving loans")
         print("UA- can be fit")
         print("U---not fit")
# Add a title
         plt.title('Pie Plot')

# Display the plot
         plt.show()
def histogram():
    df=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv",index_col=0)
    df.reset_index(drop=True, inplace=True)
    plt.hist((df['balance']), bins=5, edgecolor='black')
# Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('balance frequencies')
# Display the plot
    plt.show()
    plt.hist((df['day']), bins=5, edgecolor='black')
# Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('day frequencies')
# Display the plot
    plt.show()
def sklearn():
       df = pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\imdb data.csv")
       df_binary = df[['IMDB_Rating', 'Meta_score']]

# Taking only the selected two attributes from the dataset
       df_binary.columns = ['IMDB_Rating', 'Meta_score']
#display the first 5 rows
       df_binary.head()    
       #plotting the Scatter plot to check relationship between Sal and Temp
       sns.lmplot(x ="IMDB_Rating", y ="Meta_score", data = df_binary, order = 2, ci = None)
       plt.show()
       # Eliminating NaN or missing input numbers
       df_binary.fillna(method='ffill', inplace=True)
       X = np.array(df_binary['IMDB_Rating']).reshape(-1, 1)
       y = np.array(df_binary['Meta_score']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array
# since each dataframe contains only one column
       df_binary.dropna(inplace = True)

# Dropping any rows with Nan values
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
       regr = LinearRegression()

       regr.fit(X_train, y_train)
       print(regr.score(X_test, y_test))
       

# Main program loop
def kmeans1():
       # Read the CSV file
    df = pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\imdb data.csv")
    df1=pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")

    # Extract the desired feature for clustering
    X = df[['IMDB_Rating']].values

    # Specify the number of clusters
    k = 2

    # Create a KMeans object
    kmeans = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans.fit(X)

    # Predict the cluster labels
    labels = kmeans.predict(X)

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # Visualize the clusters
    plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], np.zeros_like(centers[:, 0]), marker='x', color='red')
    plt.title('K-means Clustering of the duration')
    plt.xlabel('balance data')
    plt.show()
    confirmed_cases = df1['balance']
    deaths = df1['transaction']
    plt.scatter(confirmed_cases, deaths)
    X = confirmed_cases.values.reshape(-2, 1)
    y = deaths.values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    plt.plot(X, y_pred, color='red', linewidth=3)
    plt.xlabel('balance')
    plt.ylabel('transaction')
    plt.title('Linear Regression: balance vs transaction')
    plt.show()


def mean_():
    print()
    mean = pf['balance'].mean()

# Calculate the median
    median = pf['balance'].median()

# Calculate the mode
    mode = pf['balance'].mode().iloc[0]

# Print the results
    print("Mean:")
    print(mean)
    print("\nMedian:")
    print(median)
    print("\nMode:")
    print(mode)
def search_name():
    # Read the CSV file
    df = pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")
    name=input()
    # Check if name is present in the DataFrame
    if name in df['name'].values:
        # Filter the DataFrame for the given name
        result = df[df['name'] == name]
        # Print the complete information for the name
        print(result)
    else:
        print("Name not found in the CSV.")

# Call the function with the name to search
def seaborn():
          df = pd.read_csv("C:\\Users\\Nirmal chaturvedi\\Desktop\\imp\\eds_mini\\bank_mangement.csv")

# Extract the x and y variables
          x = df["balance"]
          y = df["duration"]

# Create a scatter plot using seaborn
          sns.scatterplot(x=x, y=y)

# Set the plot title and labels
          plt.title("Scatter Plot")
          plt.xlabel("X")
          plt.ylabel("Y")

# Show the plot
          plt.show()
while True:
    print("1. Deposit")
    print("2. Withdraw")
    print("3. Display Transactions")
    print("4. Exit")
    print("5 for minimum balance")
    print("6 for  max balance")
    print("7 new bank account")
    print("8 to sort the balance amount ")
    print("9 group the marital data")
    print("10 group month data")
    print("11 to find the names that start with 'S' ")
    print("12 to get balance above")
    print("13 to plot the bar plot ")
    print("14 group by education")
    print("15 to group by pout_come")
    print("16 to get the duration more")
    print("17 to get the pieplot")
    print("18 to the name and the information")
    print("19 to get the transaction")
    print("20 to remove the bank account")
    print("21 to display the histogram")
    print("22 to display the regression")
    print("23 to display the prediction by kmeans")
    print("24 to describe the dataset")
    print("25 to access a particular account")
    print("26 to desplay the mean,medean and mode")
    print("27 to search particular name from the data")
    print("28 to make the scatter plot between deposie and balance")
    print("29 for the age vs duration graph")
    choice = int(input("Enter your choice: "))

    if choice == 1:
        deposit()
    elif choice == 2:
        withdraw()
    elif choice == 3:
        display_transactions()
    elif choice == 4:
        print("Exiting the program.")
        break
    elif choice == 5:
         minbal()
    elif choice == 6:
          maxbal()
    elif choice == 7:
         new_bankacc()
    elif  choice== 8:
         sort_names()
    elif choice== 9:
          group_marital()         
    elif choice== 10:
          group_month()
    elif choice== 11:
         namestart_with_s()
    elif choice== 12:
         get_balanceabove()
    elif choice== 13:
        barplot()
    elif choice== 14:
      groupby_education()
    elif choice== 15:
        groupby_poutcome()
    elif choice== 16:
         duration_more()
    elif choice== 17:
         pieplot()
    elif choice== 18:
      findname()
    elif choice== 19:
       transactions()
    elif choice== 20:
        removebank()
    elif choice==21 :
        histogram()
    elif choice==22:
        sklearn()
    elif choice==23:
        kmeans1()
    elif choice==24:
        describe()
    elif choice==25:
        access()
    elif choice==26:
        mean_()
    elif choice==27:
        search_name()
    elif choice==28:
        seaborn()
    elif choice==29:
        line()
        
    else:
        print("Invalid choice. Please try again.")
print(" thank you for using the services")
print("_/\_")