import csv

with open('Resultados_completos.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader) # Convert the reader object to a list

sorted_data = sorted(data, key=lambda row: row[0])

with open('sorted_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sorted_data)
