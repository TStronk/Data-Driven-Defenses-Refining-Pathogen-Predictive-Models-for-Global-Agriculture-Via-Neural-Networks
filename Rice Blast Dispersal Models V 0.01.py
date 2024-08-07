#!/usr/bin/env python
# coding: utf-8

# # Load in data

# In[13]:


get_ipython().system('pip install geopy')
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import L2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Load the datasets
labels_df = pd.read_csv('/Users/tedhu/OneDrive/Documents/PLP4932_First_Project/2labels.csv', encoding='ISO-8859-1')
rain_df = pd.read_csv('/Users/tedhu/OneDrive/Documents/PLP4932_First_Project/1rain.csv', encoding='ISO-8859-1')
temp_df = pd.read_csv('/Users/tedhu/OneDrive/Documents/PLP4932_First_Project/1temp.csv', encoding='ISO-8859-1')
trade_df = pd.read_csv('/Users/tedhu/OneDrive/Documents/PLP4932_First_Project/1trade.csv', encoding='ISO-8859-1')


k_temp = 1  # Constant for temperature modifier
k_precip = 1  # Constant for precipitation modifier
optimal_temp = 26.5  # Optimal temperature
optimal_precip = 0  # Optimal precipitation
attenuation = 2  # Attenuation value

# Country coordinates 
country_coordinates = {
    "Bulgaria": (42.7339, 25.4858),
    "Croatia": (45.1, 15.2),
    "France": (46.2276, 2.2137),
    "Greece": (39.0742, 21.8243),
    "Hungary": (47.1625, 19.5033),
    "Italy": (41.8719, 12.5674),
    "Portugal": (39.3999, -8.2245),
    "Romania": (45.9432, 24.9668),
    "Russia": (61.5240, 105.3188),
    "Spain": (40.4637, -3.7492),
    "Ukraine": (48.3794, 31.1656),
    "Afghanistan": (33.9391, 67.7100),
    "Bangladesh": (23.6850, 90.3563),
    "Bhutan": (27.5142, 90.4336),
    "Brunei Darussalam": (4.5353, 114.7277),
    "Cambodia": (12.5657, 104.9910),
    "China": (35.8617, 104.1954),
    "Georgia": (42.3154, 43.3569),
    "India": (20.5937, 78.9629),
    "Indonesia": (0.7893, 113.9213),
    "Iran": (32.4279, 53.6880),
    "Iraq": (33.2232, 43.6793),
    "Japan": (36.2048, 138.2529),
    "Kazakhstan": (48.0196, 66.9237),
    "Korea Republic": (35.9078, 127.7669),
    "Laos": (19.8563, 102.4955),
    "Malaysia": (4.2105, 101.9758),
    "Myanmar": (21.9162, 95.9560),
    "Nepal": (28.3949, 84.1240),
    "Pakistan": (30.3753, 69.3451),
    "Philippines": (12.8797, 121.7740),
    "Sri Lanka": (7.8731, 80.7718),
    "Taiwan": (23.6978, 120.9605),
    "Thailand": (15.8700, 100.9925),
    "Turkey": (38.9637, 35.2433),
    "Uzbekistan": (41.3775, 64.5853),
    "Vietnam": (14.0583, 108.2772),
    "Angola": (11.2027, 17.8739),
    "Benin": (9.3077, 2.3158),
    "Burkina Faso": (12.2383, -1.5616),
    "Burundi": (3.3731, 29.9189),
    "Cameroon": (7.3697, 12.3547),
    "Chad": (15.4542, 18.7322),
    "Congo": (0.2280, 15.8277),
    "Cote d'Ivoire": (7.5399, -5.5471),
    "Egypt": (26.8206, 30.8025),
    "Ethiopia": (9.1450, 40.4897),
    "Gabon": (0.8037, 11.6094),
    "Gambia": (13.4432, -15.3101),
    "Ghana": (7.9465, -1.0232),
    "Guinea": (9.9456, -9.6966),
    "Kenya": (1.2921, 36.8219),
    "Liberia": (6.4281, -9.4295),
    "Madagascar": (18.7669, 46.8691),
    "Malawi": (13.2543, 34.3015),
    "Mali": (17.5707, -3.9962),
    "Mauritius": (20.3484, 57.5522),
    "Morocco": (31.7917, -7.0926),
    "Mozambique": (18.6657, 35.5296),
    "Niger": (17.6078, 8.0817),
    "Nigeria": (9.0820, 8.6753),
    "Senegal": (14.4974, -14.4524),
    "Sierra Leone": (8.4606, -11.7799),
    "South Africa": (30.5595, 22.9375),
    "Sudan": (12.8628, 30.2176),
    "Tanzania": (6.3690, 34.8888),
    "Togo": (8.6195, 0.8248),
    "Uganda": (1.3733, 32.2903),
    "Zambia": (13.1339, 27.8493),
    "Zimbabwe": (19.0154, 29.1549),
    "Mexico": (23.6345, -102.5528),
    "United States of America": (37.0902, -95.7129),
    "Belize": (17.1899, -88.4976),
    "Costa Rica": (9.7489, -83.7534),
    "Cuba": (21.5218, -77.7812),
    "Dominican Republic": (18.7357, -70.1627),
    "El Salvador": (13.7942, -88.8965),
    "Guatemala": (15.7835, -90.2308),
    "Haiti": (18.9712, -72.2852),
    "Honduras": (15.2000, -86.2419),
    "Jamaica": (18.1096, -77.2975),
    "Nicaragua": (12.8654, -85.2072),
    "Panama": (8.5370, -80.7821),
    "Puerto Rico": (18.2208, -66.5901),
    "Trinidad and Tobago": (10.6918, -61.2225),
    "Argentina": (38.4161, -63.6167),
    "Brazil": (14.2350, -51.9253),
    "Colombia": (4.5709, -74.2973),
    "French Guiana": (3.9339, -53.1258),
    "Guyana": (4.8604, -58.9302),
    "Paraguay": (23.4425, -58.4438),
    "Peru": (9.1900, -75.0152),
    "Suriname": (3.9193, -56.0278),
    "Uruguay": (32.5228, -55.7658),
    "Venezuela": (6.4238, -66.5897),
    "Australia": (25.2744, 133.7751),
    "Fiji": (17.7134, 178.0650),
    "New Caledonia": (20.9043, 165.6180),
    "Papua New Guinea": (6.3140, 143.9555)
}


# In[5]:


# Feature extraction
def extract_features(row):
    country = row['Country']
    country = country.replace("'", "\\'")
    year = row['Year First Detected']
    temp = temp_df.query(f"Country == '{country}' and Year == {year}")['Annual mean temp (C)'].values
    precip = rain_df.query(f"Country == '{country}' and Year == {year}")['Average precipitation (mm)'].values
    trade_sum = trade_df.query(f"`Reporter Countries` == '{country}' and Year == {year}")['Tons'].sum()

    temp = temp[0] if len(temp) > 0 else 0
    precip = precip[0] if len(precip) > 0 else 0
    coords = country_coordinates.get(country, (0, 0))

    return [temp, precip, trade_sum, coords[0], coords[1]]

# Apply feature extraction
features = labels_df.apply(extract_features, axis=1, result_type='expand')
features.columns = ['Temperature', 'Precipitation', 'Trade Quantity', 'Latitude', 'Longitude']

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Prepare labels
labels = labels_df['Pest Extent'].values

# Ensure labels are 0-indexed
labels -= labels.min()
num_classes = 7

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(features_scaled, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(100, activation='relu', input_shape=(,7)),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# Output classification accuracy
print(f"Training Classification Accuracy: {train_accuracy:.2f}")
print(f"Validation Classification Accuracy: {val_accuracy:.2f}")
print(f"Test Classification Accuracy: {test_accuracy:.2f}")

# Predicting with the model
y_train_pred = model.predict(X_train).argmax(axis=1)
y_val_pred = model.predict(X_val).argmax(axis=1)
y_test_pred = model.predict(X_test).argmax(axis=1)

# Additional evaluation metrics
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importance_means = perm_importance.importances_mean
importance_std = perm_importance.importances_std

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.barh(features.columns, importance_means, xerr=importance_std, align='center')
plt.xlabel('Permutation Importance')
plt.title('Feature Importance - Neural Network')
plt.grid(True)
plt.show()

# Plotting classification lines
plt.figure(figsize=(12, 8))
plt.scatter(X_train[:, 0], y_train, color='blue', alpha=0.5, label='Actual')
plt.scatter(X_train[:, 0], y_train_pred, color='red', alpha=0.5, label='Predicted', marker='x')
plt.title('Actual vs Predicted - Training Set')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent')
plt.legend()
plt.grid(True)
plt.show()


# # Weather-Model

# In[ ]:


# Calculate standard deviation for temperature
temperature_std = temp_df['Annual mean temp (C)'].std()

# Define temperature and precipitation modifiers with std
def temp_modifier(temperature):
    return np.exp(-k_temp * ((temperature - optimal_temp)**2) / (2 * temperature_std**2))

def precip_modifier(precipitation):
    return 1 / (1 + np.exp(-k_precip * (precipitation - optimal_precip))) 

# Filter the datasets to only include years from 1990 to 2020
labels_df = labels_df[(labels_df['Year First Detected'] >= 1990) & (labels_df['Year First Detected'] <= 2020)]
rain_df = rain_df[(rain_df['Year'] >= 1990) & (rain_df['Year'] <= 2020)]
temp_df = temp_df[(temp_df['Year'] >= 1990) & (temp_df['Year'] <= 2020)]
trade_df = trade_df[(trade_df['Year'] >= 1990) & (trade_df['Year'] <= 2020)]

# Function to calculate invasion potential 
def calculate_invasion_potential(import_country, year_detected, attenuation):
    temp = temp_df.query("Country == @import_country and Year == @year_detected")['Annual mean temp (C)'].values
    precip = rain_df.query("Country == @import_country and Year == @year_detected")['Average precipitation (mm)'].values
    
    temp_factor = temp_modifier(temp[0] if len(temp) > 0 else 0)
    precip_factor = precip_modifier(precip[0] if len(precip) > 0 else 0)

    import_country_coords = country_coordinates.get(import_country)
    ip_contributions = []
    reporter_countries = trade_df.query("Year == @year_detected")['Reporter Countries'].unique()

    for reporter_country in reporter_countries:
        pest_extent = labels_df.query("Country == @reporter_country and `Year First Detected` <= @year_detected")['Pest Extent'].values
        temp1 = temp_df.query("Country == @reporter_country and Year == @year_detected")['Annual mean temp (C)'].values
        precip1 = rain_df.query("Country == @reporter_country and Year == @year_detected")['Average precipitation (mm)'].values
        temp_factor1 = temp_modifier(temp1[0] if len(temp1) > 0 else 0)
        precip_factor1 = precip_modifier(precip1[0] if len(precip1) > 0 else 0)
        reporter_country_coords = country_coordinates.get(reporter_country)
        if import_country_coords and reporter_country_coords:
            distance = geodesic(import_country_coords, reporter_country_coords).kilometers
            ip_contribution = ((1 + temp_factor1) *(1 + precip_factor1) * (1 + (pest_extent[0] if len(pest_extent) > 0 else 0))) / (1 + distance ** attenuation if distance else 1)
            ip_contributions.append(ip_contribution)

    total_ip = (1 + temp_factor) * (1 + precip_factor) * sum(ip_contributions) 
    return total_ip

# Testing across different attenuation values
attenuation_values = [1, 2, 3, 4, 5]
best_train_accuracy = 0
best_val_accuracy = 0
best_test_accuracy = 0
best_train_att = None
best_val_att = None
best_test_att = None

for att in attenuation_values:
    labels_df['Invasion Potential'] = labels_df.apply(lambda row: calculate_invasion_potential(row['Country'], row['Year First Detected'], att), axis=1)

    # Split data into features and target
    X = labels_df[['Invasion Potential']].values
    y = labels_df['Pest Extent'].values

    y -= y.min()
    num_classes = 7

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define the ANN model
    model = Sequential([
        Dense(200, activation='relu', input_shape=(1,), kernel_regularizer=L2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(150, activation='relu', kernel_regularizer=L2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(100, activation='relu', kernel_regularizer=L2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')  # Output layer nodes equal to number of unique labels
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model's performance on the training data
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    # Evaluate the model's performance on the validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    # Evaluate the model's performance on the test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Print the accuracies
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Track the best model based on accuracy
    if train_accuracy > best_train_accuracy:
        best_train_accuracy = train_accuracy
        best_train_att = att
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_val_att = att
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_test_att = att

# Output the best attenuation value and corresponding accuracies
print(f"\nBest Training Accuracy: {best_train_accuracy:.2f} (Attenuation: {best_train_att})")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f} (Attenuation: {best_val_att})")
print(f"Best Test Accuracy: {best_test_accuracy:.2f} (Attenuation: {best_test_att})")


# In[8]:


# Output the accuracies
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# # Tri-Model

# In[115]:


# Define temperature and precipitation modifiers
def temp_modifier(temperature):
    return np.exp(-k_temp * (temperature - optimal_temp)**2) 

def precip_modifier(precipitation):
    return 1 / (1 + np.exp(-k_precip * (precipitation - optimal_precip))) 

# Function to calculate invasion potential 
def calculate_invasion_potential(import_country, year_detected):
    temp = temp_df.query("Country == @import_country and Year == @year_detected")['Annual mean temp (C)'].values
    precip = rain_df.query("Country == @import_country and Year == @year_detected")['Average precipitation (mm)'].values
    
    temp_factor = temp_modifier(temp[0] if len(temp) > 0 else 0)
    precip_factor = precip_modifier(precip[0] if len(precip) > 0 else 0)

    import_country_coords = country_coordinates.get(import_country)
    ip_contributions = []
    reporter_countries = trade_df.query("Year == @year_detected")['Reporter Countries'].unique()

    for reporter_country in reporter_countries:
        pest_extent = labels_df.query("Country == @reporter_country and `Year First Detected` <= @year_detected")['Pest Extent'].values
        trade_quantity = trade_df.query("`Reporter Countries` == @reporter_country and `Partner Countries` == @import_country and Year == @year_detected")['Tons'].sum()

        reporter_country_coords = country_coordinates.get(reporter_country)
        if import_country_coords and reporter_country_coords:
            distance = geodesic(import_country_coords, reporter_country_coords).kilometers
            ip_contribution = ((1 + trade_quantity) * (1 + (pest_extent[0] if len(pest_extent) > 0 else 0))) / (1 + distance ** attenuation if distance else 1)
            ip_contributions.append(ip_contribution)

    total_ip = (1 + temp_factor) * (1 + precip_factor) * sum(ip_contributions) 
    return total_ip

# Calculate IPs for all countries in labels
labels_df['Invasion Potential'] = labels_df.apply(lambda row: calculate_invasion_potential(row['Country'], row['Year First Detected']), axis=1)

# Split data into features and target
X = labels_df[['Invasion Potential']].values
y = labels_df['Pest Extent'].values

y -= y.min()
num_classes = 7

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the ANN model
model = Sequential([
    Dense(50, activation='relu', input_shape=(1,)),
    Dense(50, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer nodes equal to number of unique labels
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model's performance on the training data
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
# Evaluate the model's performance on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
# Evaluate the model's performance on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)



# In[116]:


# Output the accuracies
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# In[117]:


sns.boxplot(x=labels_df['Pest Extent'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Pest Extent')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.show()


# In[118]:


plt.figure(figsize=(10, 6))
sns.boxplot(x=labels_df['Year First Detected'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.xticks(rotation=45)  
plt.show()


# In[119]:


# Plotting Invasion Potential for the entire dataset
plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels_df)), labels_df['Invasion Potential'], alpha=0.6)
plt.title('Scatter Plot of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.xlabel('Index')
plt.show()


# In[120]:


# Plotting Invasion Potential by Year
plt.figure(figsize=(10, 6))
for year in sorted(labels_df['Year First Detected'].unique()):
    subset = labels_df[labels_df['Year First Detected'] == year]
    plt.scatter([year] * len(subset), subset['Invasion Potential'], alpha=0.6, label=str(year))

plt.title('Scatter Plot of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[121]:


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


correct_indices = np.nonzero(y_pred_classes == y_test)[0]
misclassified_indices = np.nonzero(y_pred_classes != y_test)[0]


sorted_indices = np.argsort(X_test.flatten())
sorted_X_test = X_test.flatten()[sorted_indices]
sorted_y_pred_classes = y_pred_classes[sorted_indices]

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(X_test[correct_indices], y_test[correct_indices], color='green', label='Correctly Classified', alpha=0.5, edgecolor='k')
plt.scatter(X_test[misclassified_indices], y_test[misclassified_indices], color='red', label='Misclassified', alpha=0.5, marker='x')


plt.title('Classification Results on Test Data with Decision Boundaries')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent (True Class)')
plt.legend()
plt.grid(True)
plt.show()


cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # Duo-Model

# In[122]:


def calculate_invasion_potential_duo(import_country, year_detected):

    import_country_coords = country_coordinates.get(import_country)
    ip_contributions = []
    reporter_countries = trade_df.query("Year == @year_detected")['Reporter Countries'].unique()

    for reporter_country in reporter_countries:
        pest_extent = labels_df.query("Country == @reporter_country and `Year First Detected` <= @year_detected")['Pest Extent'].values
        trade_quantity = trade_df.query("`Reporter Countries` == @reporter_country and `Partner Countries` == @import_country and Year == @year_detected")['Tons'].sum()

        reporter_country_coords = country_coordinates.get(reporter_country)
        if import_country_coords and reporter_country_coords:
            distance = geodesic(import_country_coords, reporter_country_coords).kilometers
            ip_contribution = ((1 + trade_quantity) * (1 + (pest_extent[0] if len(pest_extent) > 0 else 0))) / (distance ** attenuation if distance else 1)
            ip_contributions.append(ip_contribution)

    total_ip =  sum(ip_contributions) if ip_contributions else 0
    return total_ip


labels_df['Invasion Potential'] = labels_df.apply(lambda row: calculate_invasion_potential_duo(row['Country'], row['Year First Detected']), axis=1)

# Split data into features and target
X = labels_df[['Invasion Potential']].values
y = labels_df['Pest Extent'].values

y -= y.min()
num_classes = 7

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the ANN model
model = Sequential([
    Dense(50, activation='relu', input_shape=(1,)),
    Dense(50, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer nodes equal to number of unique labels
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model's performance on the training data
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
# Evaluate the model's performance on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
# Evaluate the model's performance on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)


# In[123]:


# Output the accuracies
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# In[124]:


sns.boxplot(x=labels_df['Pest Extent'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Pest Extent')
plt.show()

# Plotting Invasion Potential for the entire dataset
plt.figure(figsize=(10, 6))
sns.boxplot(y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.show()


# In[125]:


# Plotting Invasion Potential by Year
plt.figure(figsize=(10, 6))
sns.boxplot(x=labels_df['Year First Detected'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.xticks(rotation=45)  # Rotating x-labels for better visibility
plt.show()


# In[126]:


plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels_df)), labels_df['Invasion Potential'], alpha=0.6)
plt.title('Scatter Plot of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.xlabel('Index')
plt.show()


# In[127]:


plt.figure(figsize=(10, 6))
for year in sorted(labels_df['Year First Detected'].unique()):
    subset = labels_df[labels_df['Year First Detected'] == year]
    plt.scatter([year] * len(subset), subset['Invasion Potential'], alpha=0.6, label=str(year))

plt.title('Scatter Plot of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[128]:


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


correct_indices = np.nonzero(y_pred_classes == y_test)[0]
misclassified_indices = np.nonzero(y_pred_classes != y_test)[0]


sorted_indices = np.argsort(X_test.flatten())
sorted_X_test = X_test.flatten()[sorted_indices]
sorted_y_pred_classes = y_pred_classes[sorted_indices]


plt.figure(figsize=(12, 8))
plt.scatter(X_test[correct_indices], y_test[correct_indices], color='green', label='Correctly Classified', alpha=0.5, edgecolor='k')
plt.scatter(X_test[misclassified_indices], y_test[misclassified_indices], color='red', label='Misclassified', alpha=0.5, marker='x')



plt.title('Classification Results on Test Data with Decision Boundaries')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent (True Class)')
plt.legend()
plt.grid(True)
plt.show()


cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # Single - Model

# In[129]:


# Function to calculate invasion potential
def calculate_invasion_potential(import_country, year_detected):

    import_country_coords = country_coordinates.get(import_country)
    ip_contributions = []
    reporter_countries = trade_df.query("Year == @year_detected")['Reporter Countries'].unique()

    for reporter_country in reporter_countries:
        pest_extent = labels_df.query("Country == @reporter_country and `Year First Detected` <= @year_detected")['Pest Extent'].values
        trade_quantity = trade_df.query("`Reporter Countries` == @reporter_country and `Partner Countries` == @import_country and Year == @year_detected")['Tons'].sum()

        reporter_country_coords = country_coordinates.get(reporter_country)
        if import_country_coords and reporter_country_coords:
            distance = geodesic(import_country_coords, reporter_country_coords).kilometers
            ip_contribution = ((1 + (pest_extent[0] if len(pest_extent) > 0 else 0))) / (distance ** attenuation if distance else 1)
            ip_contributions.append(ip_contribution)

    total_ip =  sum(ip_contributions) if ip_contributions else 0
    return total_ip

# Calculate IPs for all countries in labels
labels_df['Invasion Potential'] = labels_df.apply(lambda row: calculate_invasion_potential(row['Country'], row['Year First Detected']), axis=1)

# Split data into features and target
X = labels_df[['Invasion Potential']].values
y = labels_df['Pest Extent'].values

y -= y.min()
num_classes = 7

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the ANN model
model = Sequential([
    Dense(50, activation='relu', input_shape=(1,)),
    Dense(50, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer nodes equal to number of unique labels
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model's performance on the training data
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
# Evaluate the model's performance on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
# Evaluate the model's performance on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)


# In[130]:


# Output the accuracies
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# In[131]:


sns.boxplot(x=labels_df['Pest Extent'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Pest Extent')
plt.show()

# Plotting Invasion Potential for the entire dataset
plt.figure(figsize=(10, 6))
sns.boxplot(y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.show()


# In[132]:


# Plotting Invasion Potential by Year
plt.figure(figsize=(10, 6))
sns.boxplot(x=labels_df['Year First Detected'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.xticks(rotation=45)  # Rotating x-labels for better visibility
plt.show()


# In[133]:


# Plotting Invasion Potential for the entire dataset
plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels_df)), labels_df['Invasion Potential'], alpha=0.6)
plt.title('Scatter Plot of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.xlabel('Index')
plt.show()


# In[134]:


# Plotting Invasion Potential by Year
plt.figure(figsize=(10, 6))
for year in sorted(labels_df['Year First Detected'].unique()):
    subset = labels_df[labels_df['Year First Detected'] == year]
    plt.scatter([year] * len(subset), subset['Invasion Potential'], alpha=0.6, label=str(year))

plt.title('Scatter Plot of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[135]:


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


correct_indices = np.nonzero(y_pred_classes == y_test)[0]
misclassified_indices = np.nonzero(y_pred_classes != y_test)[0]


sorted_indices = np.argsort(X_test.flatten())
sorted_X_test = X_test.flatten()[sorted_indices]
sorted_y_pred_classes = y_pred_classes[sorted_indices]


plt.figure(figsize=(12, 8))
plt.scatter(X_test[correct_indices], y_test[correct_indices], color='green', label='Correctly Classified', alpha=0.5, edgecolor='k')
plt.scatter(X_test[misclassified_indices], y_test[misclassified_indices], color='red', label='Misclassified', alpha=0.5, marker='x')


plt.title('Classification Results on Test Data with Decision Boundaries')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent (True Class)')
plt.legend()
plt.grid(True)
plt.show()


cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # Tri-Model (Regression)

# In[87]:


import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import backend as K



# Define modifiers for temperature and precipitation
def temp_modifier(temperature):
    return np.exp(-k_temp * (temperature - optimal_temp)**2)

def precip_modifier(precipitation):
    return 1 / (1 + np.exp(-k_precip * (precipitation - optimal_precip)))

# Calculate invasion potential
def calculate_invasion_potential(import_country, year_detected):
    temp = temp_df.query("Country == @import_country and Year == @year_detected")['Annual mean temp (C)'].values
    precip = rain_df.query("Country == @import_country and Year == @year_detected")['Average precipitation (mm)'].values
    temp_factor = temp_modifier(temp[0] if len(temp) > 0 else 0)
    precip_factor = precip_modifier(precip[0] if len(precip) > 0 else 0)
    import_country_coords = country_coordinates.get(import_country)
    ip_contributions = []
    reporter_countries = trade_df.query("Year == @year_detected")['Reporter Countries'].unique()

    for reporter_country in reporter_countries:
        pest_extent = labels_df.query("Country == @reporter_country and `Year First Detected` <= @year_detected")['Pest Extent'].values
        trade_quantity = trade_df.query("`Reporter Countries` == @reporter_country and `Partner Countries` == @import_country and Year == @year_detected")['Tons'].sum()
        reporter_country_coords = country_coordinates.get(reporter_country)
        if import_country_coords and reporter_country_coords:
            distance = geodesic(import_country_coords, reporter_country_coords).kilometers
            ip_contribution = ((1 + trade_quantity) * (1 + (pest_extent[0] if len(pest_extent) > 0 else 0))) / (1 + distance ** attenuation if distance else 1)
            ip_contributions.append(ip_contribution)
    total_ip = (1 + temp_factor) * (1 + precip_factor) * sum(ip_contributions)
    return total_ip

labels_df['Invasion Potential'] = labels_df.apply(lambda row: calculate_invasion_potential(row['Country'], row['Year First Detected']), axis=1)

# Prepare data for regression model
X = labels_df[['Invasion Potential']].values
y = labels_df['Pest Extent'].values.astype(float)  

# Splitting data
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Model architecture
model = Sequential([
    Dense(100, activation='relu', input_shape=(1,)),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1) 
])

# Custom metric for Mean Error
def mean_error(y_true, y_pred):
    return K.mean(y_pred - y_true)

# Compiling the model
model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error', mean_error])

# Training model
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))

# Evaluating the model
train_mse, train_mae, train_me = model.evaluate(X_train, y_train, verbose=0)
val_mse, val_mae, val_me = model.evaluate(X_val, y_val, verbose=0)
test_mse, test_mae, test_me = model.evaluate(X_test, y_test, verbose=0)


# In[88]:


# Output results
print(f"Training - MSE: {train_mse:.2f}, MAE: {train_mae:.2f}, ME: {train_me:.2f}")
print(f"Validation - MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, ME: {val_me:.2f}")
print(f"Test - MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, ME: {test_me:.2f}")


# In[89]:


sns.boxplot(x=labels_df['Pest Extent'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Pest Extent')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.show()


# In[90]:


plt.figure(figsize=(10, 6))
sns.boxplot(x=labels_df['Year First Detected'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.xticks(rotation=45) 
plt.show()


# In[91]:


plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels_df)), labels_df['Invasion Potential'], alpha=0.6)
plt.title('Scatter Plot of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.xlabel('Index')
plt.show()


# In[92]:


plt.figure(figsize=(10, 6))
for year in sorted(labels_df['Year First Detected'].unique()):
    subset = labels_df[labels_df['Year First Detected'] == year]
    plt.scatter([year] * len(subset), subset['Invasion Potential'], alpha=0.6, label=str(year))
plt.title('Scatter Plot of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[93]:


import matplotlib.pyplot as plt
import seaborn as sns


y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)


train_results = np.hstack((X_train, y_train.reshape(-1,1), y_pred_train))
val_results = np.hstack((X_val, y_val.reshape(-1,1), y_pred_val))
test_results = np.hstack((X_test, y_test.reshape(-1,1), y_pred_test))

all_results = np.vstack((train_results, val_results, test_results))
all_results_df = pd.DataFrame(all_results, columns=['Invasion Potential', 'Actual Pest Extent', 'Predicted Pest Extent'])

sorted_df = all_results_df.sort_values(by='Invasion Potential')

plt.figure(figsize=(12, 8))
sns.scatterplot(data=all_results_df, x='Invasion Potential', y='Actual Pest Extent', color='blue', label='Actual Data Points')
sns.lineplot(data=sorted_df, x='Invasion Potential', y='Predicted Pest Extent', color='red', label='Regression Line')
plt.title('Actual vs. Predicted Pest Extent')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent')
plt.legend()
plt.grid(True)
plt.show()


# # Rounded Regression as Classification

# In[94]:


y_train_pred = model.predict(X_train).flatten()  
y_val_pred = model.predict(X_val).flatten()
y_test_pred = model.predict(X_test).flatten()

# Rounding the predictions to nearest integer for classification
y_train_pred_rounded = np.rint(y_train_pred)
y_val_pred_rounded = np.rint(y_val_pred)
y_test_pred_rounded = np.rint(y_test_pred)

# Evaluating classification accuracy
train_accuracy = accuracy_score(y_train, y_train_pred_rounded)
val_accuracy = accuracy_score(y_val, y_val_pred_rounded)
test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

# Output classification accuracy
print(f"Training Classification Accuracy: {train_accuracy:.2f}")
print(f"Validation Classification Accuracy: {val_accuracy:.2f}")
print(f"Test Classification Accuracy: {test_accuracy:.2f}")

# Additional evaluation metrics
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred_rounded))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred_rounded)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[95]:


plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', label='Actual', alpha=0.5, edgecolor='k')
plt.scatter(X_train, y_train_pred_rounded, color='red', label='Predicted', alpha=0.5, marker='x')

sorted_indices = np.argsort(X_train.flatten())
sorted_X_train = X_train.flatten()[sorted_indices]
sorted_y_pred_rounded = y_train_pred_rounded[sorted_indices]

for i in range(len(sorted_X_train)-1):
    if sorted_y_pred_rounded[i] != sorted_y_pred_rounded[i+1]:
        plt.axvline(x=sorted_X_train[i], color='k', linestyle='--', lw=1)

plt.title('Actual vs Predicted Training Data with Decision Boundaries')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent (Rounded Prediction)')
plt.legend()
plt.grid(True)
plt.show()


# # Duo-Model (Regression)

# In[96]:


# Calculate invasion potential
def calculate_invasion_potential(import_country, year_detected):
    temp = temp_df.query("Country == @import_country and Year == @year_detected")['Annual mean temp (C)'].values
    precip = rain_df.query("Country == @import_country and Year == @year_detected")['Average precipitation (mm)'].values
    temp_factor = temp_modifier(temp[0] if len(temp) > 0 else 0)
    precip_factor = precip_modifier(precip[0] if len(precip) > 0 else 0)
    import_country_coords = country_coordinates.get(import_country)
    ip_contributions = []
    reporter_countries = trade_df.query("Year == @year_detected")['Reporter Countries'].unique()

    for reporter_country in reporter_countries:
        pest_extent = labels_df.query("Country == @reporter_country and `Year First Detected` <= @year_detected")['Pest Extent'].values
        trade_quantity = trade_df.query("`Reporter Countries` == @reporter_country and `Partner Countries` == @import_country and Year == @year_detected")['Tons'].sum()
        reporter_country_coords = country_coordinates.get(reporter_country)
        if import_country_coords and reporter_country_coords:
            distance = geodesic(import_country_coords, reporter_country_coords).kilometers
            ip_contribution = ((1 + trade_quantity) * (1 + (pest_extent[0] if len(pest_extent) > 0 else 0))) / (1 + distance ** attenuation if distance else 1)
            ip_contributions.append(ip_contribution)
    total_ip =  sum(ip_contributions)
    return total_ip

labels_df['Invasion Potential'] = labels_df.apply(lambda row: calculate_invasion_potential(row['Country'], row['Year First Detected']), axis=1)

# Prepare data for regression model
X = labels_df[['Invasion Potential']].values
y = labels_df['Pest Extent'].values.astype(float)  

# Splitting data
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Model architecture
model = Sequential([
    Dense(100, activation='relu', input_shape=(1,)),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1)  # Single output node for regression
])

# Custom metric for Mean Error
def mean_error(y_true, y_pred):
    return K.mean(y_pred - y_true)

# Compiling the model
model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error', mean_error])

# Training model
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))

# Evaluating the model
train_mse, train_mae, train_me = model.evaluate(X_train, y_train, verbose=0)
val_mse, val_mae, val_me = model.evaluate(X_val, y_val, verbose=0)
test_mse, test_mae, test_me = model.evaluate(X_test, y_test, verbose=0)




# In[97]:


# Output results
print(f"Training - MSE: {train_mse:.2f}, MAE: {train_mae:.2f}, ME: {train_me:.2f}")
print(f"Validation - MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, ME: {val_me:.2f}")
print(f"Test - MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, ME: {test_me:.2f}")


# In[98]:


sns.boxplot(x=labels_df['Pest Extent'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Pest Extent')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.show()


# In[99]:


plt.figure(figsize=(10, 6))
sns.boxplot(x=labels_df['Year First Detected'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.xticks(rotation=45) 
plt.show()


# In[100]:


plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels_df)), labels_df['Invasion Potential'], alpha=0.6)
plt.title('Scatter Plot of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.xlabel('Index')
plt.show()


# In[101]:


plt.figure(figsize=(10, 6))
for year in sorted(labels_df['Year First Detected'].unique()):
    subset = labels_df[labels_df['Year First Detected'] == year]
    plt.scatter([year] * len(subset), subset['Invasion Potential'], alpha=0.6, label=str(year))
plt.title('Scatter Plot of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[102]:


y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)


train_results = np.hstack((X_train, y_train.reshape(-1,1), y_pred_train))
val_results = np.hstack((X_val, y_val.reshape(-1,1), y_pred_val))
test_results = np.hstack((X_test, y_test.reshape(-1,1), y_pred_test))

all_results = np.vstack((train_results, val_results, test_results))
all_results_df = pd.DataFrame(all_results, columns=['Invasion Potential', 'Actual Pest Extent', 'Predicted Pest Extent'])

sorted_df = all_results_df.sort_values(by='Invasion Potential')

plt.figure(figsize=(12, 8))
sns.scatterplot(data=all_results_df, x='Invasion Potential', y='Actual Pest Extent', color='blue', label='Actual Data Points')
sns.lineplot(data=sorted_df, x='Invasion Potential', y='Predicted Pest Extent', color='red', label='Regression Line')
plt.title('Actual vs. Predicted Pest Extent')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent')
plt.legend()
plt.grid(True)
plt.show()


# # Rounded Regression as Classification

# In[103]:


y_train_pred = model.predict(X_train).flatten()  
y_val_pred = model.predict(X_val).flatten()
y_test_pred = model.predict(X_test).flatten()

y_train_pred_rounded = np.rint(y_train_pred)
y_val_pred_rounded = np.rint(y_val_pred)
y_test_pred_rounded = np.rint(y_test_pred)

train_accuracy = accuracy_score(y_train, y_train_pred_rounded)
val_accuracy = accuracy_score(y_val, y_val_pred_rounded)
test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

print(f"Training Classification Accuracy: {train_accuracy:.2f}")
print(f"Validation Classification Accuracy: {val_accuracy:.2f}")
print(f"Test Classification Accuracy: {test_accuracy:.2f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred_rounded))

conf_matrix = confusion_matrix(y_test, y_test_pred_rounded)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[104]:


plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', label='Actual', alpha=0.5, edgecolor='k')
plt.scatter(X_train, y_train_pred_rounded, color='red', label='Predicted', alpha=0.5, marker='x')

sorted_indices = np.argsort(X_train.flatten())
sorted_X_train = X_train.flatten()[sorted_indices]
sorted_y_pred_rounded = y_train_pred_rounded[sorted_indices]

for i in range(len(sorted_X_train)-1):
    if sorted_y_pred_rounded[i] != sorted_y_pred_rounded[i+1]:
        plt.axvline(x=sorted_X_train[i], color='k', linestyle='--', lw=1)

plt.title('Actual vs Predicted Training Data with Decision Boundaries')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent (Rounded Prediction)')
plt.legend()
plt.grid(True)
plt.show()


# # Single-Model

# In[105]:


# Calculate invasion potential
def calculate_invasion_potential(import_country, year_detected):
    temp = temp_df.query("Country == @import_country and Year == @year_detected")['Annual mean temp (C)'].values
    precip = rain_df.query("Country == @import_country and Year == @year_detected")['Average precipitation (mm)'].values
    temp_factor = temp_modifier(temp[0] if len(temp) > 0 else 0)
    precip_factor = precip_modifier(precip[0] if len(precip) > 0 else 0)
    import_country_coords = country_coordinates.get(import_country)
    ip_contributions = []
    reporter_countries = trade_df.query("Year == @year_detected")['Reporter Countries'].unique()

    for reporter_country in reporter_countries:
        pest_extent = labels_df.query("Country == @reporter_country and `Year First Detected` <= @year_detected")['Pest Extent'].values
        trade_quantity = trade_df.query("`Reporter Countries` == @reporter_country and `Partner Countries` == @import_country and Year == @year_detected")['Tons'].sum()
        reporter_country_coords = country_coordinates.get(reporter_country)
        if import_country_coords and reporter_country_coords:
            distance = geodesic(import_country_coords, reporter_country_coords).kilometers
            ip_contribution = ((1 + (pest_extent[0] if len(pest_extent) > 0 else 0)) / (1 + distance ** attenuation if distance else 1))
            ip_contributions.append(ip_contribution)
    total_ip =  sum(ip_contributions)
    return total_ip

labels_df['Invasion Potential'] = labels_df.apply(lambda row: calculate_invasion_potential(row['Country'], row['Year First Detected']), axis=1)

# Prepare data for regression model
X = labels_df[['Invasion Potential']].values
y = labels_df['Pest Extent'].values.astype(float)  

# Splitting data
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Model architecture
model = Sequential([
    Dense(100, activation='relu', input_shape=(1,)),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1)  
])


# Custom metric for Mean Error
def mean_error(y_true, y_pred):
    return K.mean(y_pred - y_true)

# Compiling the model
model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error', mean_error])

# Training model
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val))

# Evaluating the model
train_mse, train_mae, train_me = model.evaluate(X_train, y_train, verbose=0)
val_mse, val_mae, val_me = model.evaluate(X_val, y_val, verbose=0)
test_mse, test_mae, test_me = model.evaluate(X_test, y_test, verbose=0)



# In[106]:


# Output results
print(f"Training - MSE: {train_mse:.2f}, MAE: {train_mae:.2f}, ME: {train_me:.2f}")
print(f"Validation - MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, ME: {val_me:.2f}")
print(f"Test - MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, ME: {test_me:.2f}")


# In[107]:


sns.boxplot(x=labels_df['Pest Extent'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Pest Extent')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.show()


# In[108]:


plt.figure(figsize=(10, 6))
sns.boxplot(x=labels_df['Year First Detected'], y=labels_df['Invasion Potential'])
plt.title('Distribution of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.xticks(rotation=45)  # Rotating x-labels for better visibility
plt.show()


# In[109]:


plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels_df)), labels_df['Invasion Potential'], alpha=0.6)
plt.title('Scatter Plot of Invasion Potential Across All Data')
plt.ylabel('Invasion Potential')
plt.xlabel('Index')
plt.show()


# In[110]:


plt.figure(figsize=(10, 6))
for year in sorted(labels_df['Year First Detected'].unique()):
    subset = labels_df[labels_df['Year First Detected'] == year]
    plt.scatter([year] * len(subset), subset['Invasion Potential'], alpha=0.6, label=str(year))
plt.title('Scatter Plot of Invasion Potential by Year')
plt.xlabel('Year First Detected')
plt.ylabel('Invasion Potential')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[111]:


y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

train_results = np.hstack((X_train, y_train.reshape(-1,1), y_pred_train))
val_results = np.hstack((X_val, y_val.reshape(-1,1), y_pred_val))
test_results = np.hstack((X_test, y_test.reshape(-1,1), y_pred_test))

all_results = np.vstack((train_results, val_results, test_results))
all_results_df = pd.DataFrame(all_results, columns=['Invasion Potential', 'Actual Pest Extent', 'Predicted Pest Extent'])

sorted_df = all_results_df.sort_values(by='Invasion Potential')

plt.figure(figsize=(12, 8))
sns.scatterplot(data=all_results_df, x='Invasion Potential', y='Actual Pest Extent', color='blue', label='Actual Data Points')
sns.lineplot(data=sorted_df, x='Invasion Potential', y='Predicted Pest Extent', color='red', label='Regression Line')
plt.title('Actual vs. Predicted Pest Extent')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent')
plt.legend()
plt.grid(True)
plt.show()


# # Rounded Regression as classification

# In[112]:


import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predicting with the model
y_train_pred = model.predict(X_train).flatten() 
y_val_pred = model.predict(X_val).flatten()
y_test_pred = model.predict(X_test).flatten()

y_train_pred_rounded = np.rint(y_train_pred)
y_val_pred_rounded = np.rint(y_val_pred)
y_test_pred_rounded = np.rint(y_test_pred)

# Evaluating classification accuracy
train_accuracy = accuracy_score(y_train, y_train_pred_rounded)
val_accuracy = accuracy_score(y_val, y_val_pred_rounded)
test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

# Output classification accuracy
print(f"Training Classification Accuracy: {train_accuracy:.2f}")
print(f"Validation Classification Accuracy: {val_accuracy:.2f}")
print(f"Test Classification Accuracy: {test_accuracy:.2f}")

# Additional evaluation metrics
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred_rounded))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred_rounded)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[113]:


plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', label='Actual', alpha=0.5, edgecolor='k')
plt.scatter(X_train, y_train_pred_rounded, color='red', label='Predicted', alpha=0.5, marker='x')

sorted_indices = np.argsort(X_train.flatten())
sorted_X_train = X_train.flatten()[sorted_indices]
sorted_y_pred_rounded = y_train_pred_rounded[sorted_indices]

for i in range(len(sorted_X_train)-1):
    if sorted_y_pred_rounded[i] != sorted_y_pred_rounded[i+1]:
        plt.axvline(x=sorted_X_train[i], color='k', linestyle='--', lw=1)

plt.title('Actual vs Predicted Training Data with Decision Boundaries')
plt.xlabel('Invasion Potential')
plt.ylabel('Pest Extent (Rounded Prediction)')
plt.legend()
plt.grid(True)
plt.show()

