# import matplotlib.pyplot as plt
# import numpy as np
#
#
# x = [5, 10, 15, 20, 25, 30]
#
#
# y = [15, 43, 60, 70, 80, 90]
#
# for i in range(10):
#     aaaaa = (y + np.random.normal(0, 5, len(y)))
#     plt.plot(x, aaaaa, marker='o', linestyle='-')
#
# # plt.plot(x, y1, marker='o', linestyle='-')
# # plt.title('Data Points Plot')
# plt.xlabel('Duration of input audio (s)')
# plt.ylabel('Computation time (s)')
# plt.grid(True)
#
# plt.show()

# import matplotlib.pyplot as plt
#
# # Data for the categories and their corresponding percentages
# categories = ['2 notes',  '3 notes', '4 notes', '5 notes']
# percentages = [96 , 88, 80, 73]  # Replace with your actual percentages
#
# # Create a bar graph
# plt.figure(figsize=(8, 6))  # Optional: Set the figure size
# plt.bar(categories, percentages, color='royalblue')
#
# # Optional: Add labels and title
# plt.xlabel('Degree of polyphony')
# plt.ylabel('F-measure')
# # plt.title('Percentage by Category')
#
# # Display the graph
# plt.show()

# import matplotlib.pyplot as plt
#
# # Data for the categories and their corresponding percentages
# categories = ['Computer lab',  'Bedroom', 'Bathroom']
# percentages = [83, 81, 77]  # Replace with your actual percentages
#
# # Create a bar graph
# plt.figure(figsize=(8, 6))  # Optional: Set the figure size
# plt.bar(categories, percentages, color='royalblue')
#
# # Optional: Add labels and title
# plt.xlabel('Recording environment')
# plt.ylabel('F-measure')
# # plt.title('Percentage by Category')
#
# # Display the graph
# plt.show()
#
# import matplotlib.pyplot as plt
#
# # Data for the x and y values
# x_values = [500, 400, 300, 200, 100]
# y_values = [100, 100, 98, 96, 95]
#
# # Create a line graph
# plt.figure(figsize=(8, 6))  # Optional: Set the figure size
# plt.plot(x_values, y_values, marker='o', linestyle='-')
#
# # Optional: Add labels and title
# plt.xlabel('Note rate (ms/note)')
# plt.ylabel('F-measure')
# # plt.title('Line Graph Example')
#
# plt.ylim(0, 100)
#
# # Display the graph
# plt.grid(True)  # Optional: Add a grid
# plt.show()


import matplotlib.pyplot as plt

# Data for the x and y values
x_values = [-80, -70, -60, -50, -40]
y_values = [85, 84, 82, 81, 78]

# Create a line graph
plt.figure(figsize=(8, 6))  # Optional: Set the figure size
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Optional: Add labels and title
plt.xlabel('Noise level (dBspl)')
plt.ylabel('F-measure')
# plt.title('Line Graph Example')

plt.ylim(0, 100)

# Display the graph
plt.grid(True)  # Optional: Add a grid
plt.show()


