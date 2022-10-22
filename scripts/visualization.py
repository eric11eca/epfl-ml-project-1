import os
from pickle import NONE
import matplotlib.pyplot as plt 
import seaborn as sns


plt.figure(figsize=(12, 8))
def plot_training_stats(stats_dict, save_path, title=None):
    print("--------------------------")
    print("| \t BEST Result \t | ")
    print("--------------------------")
    
    for res, value in stats_dict.items():
        if type(value) == list:
            print(f'{res} => {value[-1]}')
            ## Plot
            res_names_set_dict = {
                'Loss': ['train_loss', 'val_loss'],
                'Accuracy': ['train_acc', 'val_acc'],
                'Prec/Recall': ['train_precision', 'train_recall', 'val_precision', 'val_recall']
            }
            for i, (key, res_names) in enumerate(res_names_set_dict.items()):
                plt.subplot(1,3,i+1)
                for name in res_names:
                    plt.plot(res_names_set_dict[name])
                plt.legend(res_names)
                plt.title(key)

            if title is not None:
                plt.suptitle(title)
            plt.savefig(save_path)
            # plt.show()
        else:
            print(f'{res} => {value}')