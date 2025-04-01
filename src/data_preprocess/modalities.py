import pickle
import matplotlib.pyplot as plt

# Load from the .pkl file
with open("modalities.pkl", "rb") as f:  # "rb" means read binary
    loaded_data = pickle.load(f)

keys = list(loaded_data.keys())
keys.sort()
for i in keys:
    print(i)
    print(loaded_data[i])

# Extract keys and their corresponding list lengths
keys = list(loaded_data.keys())
list_lengths = [len(lst) for lst in loaded_data.values()]

# Create the bar plot
plt.figure(figsize=(10, 5))  # Set figure size
plt.bar(keys, list_lengths, color="skyblue", edgecolor="black")

# Add labels and title
plt.xlabel("Keys")
plt.ylabel("List Lengths")
plt.title("Bar Plot of List Lengths for Each Key")
plt.xticks(rotation=45, ha="right")  # Rotate keys for readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()
