import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('results_final.csv')

# Calculate averages for every 50 entries
averages = []
for i in range(0, len(data), 50):
    avg = data.iloc[i:i+50, 1].mean()
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
