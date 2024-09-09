import csv

def extract_matching_rows(file1, file2, output_file):
    # Function to convert string to float and round
    def convert_and_round(value):
        try:
            return round(float(value), 4)
        except ValueError:
            return value

    # Read and convert the first CSV file
    with open(file1, 'r') as f1:
        data1 = [[convert_and_round(cell) for cell in row] for row in csv.reader(f1)]
    
    # Read and convert the second CSV file
    with open(file2, 'r') as f2:
        data2 = [[convert_and_round(cell) for cell in row] for row in csv.reader(f2)]
    
    # Find matching rows and merge
    matching_rows = []
    for row1 in data1:
        for row2 in data2:
            if row1[:3] == row2[:3]:
                print(row1[:3])
                print(row2[:3])
                # Merge row1's first 3 columns with row2's last column
                merged_row = row1[:3] + [row2[-1]]
                matching_rows.append(merged_row)
                break
    
    # Sort matching_rows by the third column (index 2) in descending order
    matching_rows.sort(key=lambda x: x[3], reverse=True)
    
    # Write sorted matching rows to output file
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(matching_rows)

# Usage
extract_matching_rows('UoC_5.csv', 'data_points_origin.csv', 'matching_rows.csv')
print("Matching rows have been extracted, merged, sorted, and saved to 'matching_rows.csv'")