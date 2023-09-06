import matplotlib.pyplot as plt
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