import csv
filename = 'data.csv'
with open(filename, mode='w') as file:
    writer = csv.writer(file)

    writer.writerow(['Programming language', 'Designed by', 'Appeared', 'Extension'])
    writer.writerow(['Python', 'Guido, van, Rossum', '1991', '.py'])  # testing ","
    writer.writerow(['Java', 'James Gosling', '1995', '.java'])
    writer.writerow(['C++', 'Bjarne Stroustrup', '1985', '.cpp'])

# import panda as pd  # Error is raised, thank you... otherwise, i will more lost
import pandas as pd
data = pd.read_csv(filename)
print(data.head())
