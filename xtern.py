import csv
import regex
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import statsmodels.api as sm

sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:\\Users\\saumya\\Downloads\\2020-XTern-DS.csv")
data = data.replace('-', np.nan, regex=True)
data = data.replace('1,', np.nan, regex=True)
data = data.replace('NEW', np.nan, regex=True)
data = data.replace('Opening Soon', np.nan, regex=True)
data.dropna(axis=0, how='any', inplace=True)
data['Rating'] = data['Rating'].astype(float)
data['Average_Cost'] = data['Average_Cost'].str.replace('$', '')
data['Average_Cost'] = data['Average_Cost'].astype(float)
data = data[data['Average_Cost'] <= 50]
data['Cook_Time'] = data['Cook_Time'].str.replace(' minutes', '')
data['Cook_Time'] = data['Cook_Time'].astype(float)
data = data[data['Cook_Time'] <= 65]
high = data[data['Rating'] >= 3.9]
high = high[high['Average_Cost'] <= 25]
high = high[high['Cook_Time'] <= 30]
X = high.loc[:, ['Average_Cost', 'Latitude', 'Longitude', 'Cuisines']]
K_clusters = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = data[['Latitude']]
X_axis = data[['Longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

kmeans = KMeans(n_clusters=8, init='k-means++')
kmeans.fit(X[X.columns[1:3]])  # Compute k-means clustering.
X['cluster_label'] = kmeans.fit_predict(X[X.columns[1:3]])
centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
print(centers)
labels = kmeans.predict(X[X.columns[1:3]])  # Labels of each point
# KNN Cluster
X.plot.scatter(x='Latitude', y='Longitude', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
for i in range(3):
    Y = X[X['cluster_label'] == i]
    Y = Y['Cuisines']
    print(Y)
plt.show()

# LATITUDE AND LONGITUDE AND CUISINE OF RESTAURANT WITH RATING >= 3.9, AVERAGE COST <25 DOLLARS AND COOK TIME <= 30
# MINUTES

model = LinearRegression()
data['Minimum_Order'] = data['Minimum_Order'].str.replace('$', '')
data['Minimum_Order'] = data['Minimum_Order'].astype(float)
df = data['Minimum_Order']
data['Minimum_Order'] = df[abs(df - np.mean(df)) < 6 * np.std(df)]
data.dropna(axis=0, how='any', inplace=True)
X = data.loc[:, ['Cook_Time', 'Rating', 'Minimum_Order', 'Average_Cost']]
Min = np.array(data['Minimum_Order'])
Avg = X['Average_Cost']
Avg = Avg.astype(float)
Avg = np.array(Avg)
Cook = np.array(X['Cook_Time'])
All = np.vstack((X['Cook_Time'], Min, Avg)).T
X['Rating'] = X['Rating'].astype(float)
Rate = np.array(X['Rating'])
model.fit(All, Rate)
model = LinearRegression().fit(All, Rate)
est = sm.OLS(Rate, All)
est = est.fit()
print(est.summary())
slope, intercept, r, p, std_err = stats.linregress(Cook, Rate)
mymodel = slope * Cook + intercept
plt.scatter(Cook, Rate)
plt.plot(Cook, mymodel)
plt.title('COOK')
plt.show()
slope, intercept, r, p, std_err = stats.linregress(Avg, Rate)
mymodel = slope * Avg + intercept
plt.scatter(Avg, Rate)
plt.plot(Avg, mymodel)
plt.title("AVG")
plt.show()
slope, intercept, r, p, std_err = stats.linregress(Min, Rate)
mymodel = slope * Min + intercept
plt.scatter(Min, Rate)
plt.plot(Min, mymodel)
plt.title("MIN")
plt.show()

model = LinearRegression()
data.dropna(axis=0, how='any', subset=['Reviews', 'Rating', 'Votes'], inplace=True)
X = data.loc[:, ['Votes', 'Rating', 'Reviews']]
Vote = X['Votes']
Vote = Vote.astype(float)
Vote = np.array(Vote)
Rev = X['Reviews']
Rev = Rev.astype(float)
Rev = np.array(Rev)
All = np.vstack((Vote, Rev)).T
X['Rating'] = X['Rating'].astype(float)
Rate = np.array(X['Rating'])
model.fit(All, Rate)
model = LinearRegression().fit(All, Rate)
est = sm.OLS(Rate, All)
est = est.fit()
print(est.summary())

slope, intercept, r, p, std_err = stats.linregress(Avg, Cook)
mymodel = slope * Avg + intercept
plt.scatter(Avg, Cook)
plt.plot(Avg, mymodel)
plt.title("AVG and COOK")
plt.show()
model = LinearRegression()
Avg = Avg.reshape((-1,1))
model.fit(Avg, Cook)
model = LinearRegression().fit(Avg, Cook)
est = sm.OLS(Avg, Cook)
est = est.fit()
print(est.summary())
# AVERAGE COST, COOK TIME, VOTES AND REVIEWS HAVE A SIGNIFICANT IMPACT ON THE RESTAURANT RATING, MIn order does not
# affect rating that much

# With increase in Average Cost, Cook time and Minimum order there is increase in rating
