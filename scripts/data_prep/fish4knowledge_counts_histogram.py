import re
import matplotlib.pyplot as plt

# The raw data provided
raw_data = """
    01.Dascyllus reticulatus 
    12112 
    

    02.Plectroglyphidodon dickii 
    2683
    

    03.Chromis chrysura 
    3593
    

    04.Amphiprion clarkii 
    4049
    

    05.Chaetodon lunulatus 
    2534
    

    06.Chaetodon trifascialis 
    190
    

    07.Myripristis kuntee 
    450
    

    08.Acanthurus nigrofuscus 
    218
    

    09.Hemigymnus fasciatus 
    241
    

    10.Neoniphon sammara  
    299
    

    11.Abudefduf vaigiensis 
    98
    

    12.Canthigaster valentini 
    147
    

    13.Pomacentrus moluccensis 
    181
    

    14.Zebrasoma scopas 
    90
    

    15.Hemigymnus melapterus 
    42
    

    16.Lutjanus fulvus 
    206
    

    17.Scolopsis bilineata 
    49
    

    18.Scaridae  
    56
   

    19.Pempheris vanicolensis  
    29


    20.Zanclus cornutus 
    21
    

    21.Neoglyphidodon nigroris  
    16
    

    22.Balistapus undulatus  
    41
    

    23.Siganus fuscescens  
    25
    
"""

# 1. Parse the data
pattern = r"(\d{2}\.[A-Za-z\s]+?)\s+(\d+)"
matches = re.findall(pattern, raw_data)

species_names = [match[0].split('.', 1)[1].strip() for match in matches]
detection_counts = [int(match[1]) for match in matches]

# 2. Sort the data by frequency (descending order)
# Combine lists into tuples, sort based on the count (x[1]), and unzip back into lists
sorted_data = sorted(zip(species_names, detection_counts), key=lambda x: x[1], reverse=True)
species_names_sorted, detection_counts_sorted = zip(*sorted_data)

# 3. Plot the data
plt.figure(figsize=(12, 8))

# Create a horizontal bar chart with the sorted data
bars = plt.barh(species_names_sorted, detection_counts_sorted, color='skyblue', edgecolor='black')

# Invert the y-axis so the highest number species remains at the top
plt.gca().invert_yaxis()

# Add data labels to the end of each bar
for bar in bars:
    plt.text(
        bar.get_width() + 100,              
        bar.get_y() + bar.get_height() / 2, 
        f'{int(bar.get_width()):,}',        
        va='center',
        ha='left',
        fontsize=10
    )

# Formatting the plot
plt.title('Species Detection Counts (Ordered by Frequency)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Detection Count', fontsize=12)
plt.ylabel('Species', fontsize=12)
plt.margins(x=0.15) 
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.show()