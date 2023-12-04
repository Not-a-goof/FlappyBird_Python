import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'results14.csv'
data = pd.read_csv(file_path)

x = data.iloc[:, 0]
y = data.iloc[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Performance over time')
plt.xlabel('Number of runs')
plt.ylabel('Number of pipes passed')
plt.grid(True)
plt.show()
plt.savefig('performange.png')
