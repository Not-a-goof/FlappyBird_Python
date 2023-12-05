import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('results_final.csv')  # Replace with your file path

# Calculate averages for every 50 entries in the second column
averages = []
for i in range(0, len(data), 50):
    avg = data.iloc[i:i+30, 1].mean()  # Assuming second column is at index 1
    averages.append(avg)

# Plotting the averages
plt.figure(figsize=(10, 6))
plt.plot(averages, marker=',', linestyle='-')
plt.title('Average of Every 50 Entries from high score')
plt.xlabel('Episodes (Each Represents 50 Episodes)')
plt.ylabel('Average Value')
plt.grid(True)
plt.show()
plt.savefig('performange.png')
