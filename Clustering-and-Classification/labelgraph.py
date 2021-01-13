import sys
import pandas as pd
import matplotlib.pyplot as plt

# c1_original_space_labels
# c2_new_space_labels
# c3_classification_labels

print(sys.argv[1])
if len(sys.argv)>0:
    df = pd.read_csv(sys.argv[1],sep = '\t\t', engine = 'python').astype(int)
    df.columns = ['INDEX', 'LABELS', 'CORRECT_LABELS']
    correct = []
    incorrect = []
    labels = []
    for i in range(10):
        correct.append(df.loc[(df['CORRECT_LABELS'] == i) & (df["LABELS"] == i)].shape[0])
        incorrect.append(df.loc[(df['CORRECT_LABELS'] == i) & (df["LABELS"] != i)].shape[0])
        labels.append(str(i))

    df = pd.DataFrame({'Correct Labels': correct, 'Incorrect Labels': incorrect}, index=labels)
    ax = df.plot.bar(rot=0)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()))
    plt.tight_layout()
    plt.savefig(sys.argv[1]+".png")
    plt.show()
    plt.close()
