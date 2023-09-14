import matplotlib.pyplot as plt
import pandas as pd

def box_plotting(dataset, by_):
    num_columns = len(dataset.columns) - 1  
    num_rows = (num_columns + 1) // 2  

# Create subplots with the calculated layout
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 8))

# Iterate through the columns
    for i, col in enumerate(dataset.columns[:-1]):
        ax = axes[i // 2, i % 2]  
        dataset.boxplot(column=col, by=by_, ax=ax)
        ax.set_title(f'Boxplot for {col}')

    if num_columns % 2 != 0:
        fig.delaxes(axes[-1, -1])

    plt.tight_layout()
    plt.show()
    
    #The delimier is ';' instead of ','
def read_file(filename):
    return pd.read_csv(filename)
 
def changeDelimiter(filename):
    df = pd.read_csv(filename, delimiter=';')
    df = df.replace(';', ',', regex=True)
    df.to_csv(filename, index=False)
    print("File successfully changed")
    
def showInfo(dataset_name, shape=True, head=20, tail=None):
    if shape:
        print("displaying shape Information")
        print(dataset_name.shape)
    
    print("Displaying head of Dataset:")
    print(dataset_name.head(head))
    
    if tail is not None:
        print("Tail of Dataset: ")
        print(dataset_name.tail(tail))
    
    print("Displaying info of Dataset: ")
    print(dataset_name.info()) 
    
    print("Displaying Description of Dataset:")
    print(dataset_name.describe())  
    
    return
#%%
