# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt 


def show_polar_heat_map(polar_grads, dim_names, text, max_words = 100, width=16, height=4, save_path=''):
    plt.figure(figsize=(width, height))
    df_polar = pd.DataFrame(polar_grads)
    df_polar = df_polar.rename(index= dict(enumerate(text)), columns=dict(enumerate([y+'/'+x for x,y in dim_names])))
    seaborn.heatmap(df_polar[:max_words])
    if save_path:
        plt.savefig('../../data/plots/'+save_path)
    else:
        plt.show()

