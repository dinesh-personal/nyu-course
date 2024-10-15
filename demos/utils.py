import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder

#import pandas as pd


def plot_data(x_values, y_values):
    plt.figure(figsize=(6, 2))    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(x_values, y_values,c='b', marker=".") 

def plot_3d(x_values, y_values, z_values, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_values, y_values, z_values, marker='o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

def read_data(file_name, x_cols=1):
    data = np.genfromtxt(file_name, dtype=float, delimiter=',', names=True)
    if x_cols == 1:
        X = np.array([x[0] for x in data])
        Y = np.array([x[1] for x in data])
        return X, Y
    else:          
        X = np.array([[x[i] for i in range(x_cols)] for x in data])
        Y = np.array([x[x_cols] for x in data])
        return X, Y


def create_color_dict(unique_colors, color_names):
    c_length = len(color_names)
    color_dict = dict()
    for index,c in enumerate(unique_colors):
        name = c
        value = color_names[index % c_length]
        color_dict[name] = value
    return color_dict
    

def create_color_mapping(unique_colors):
    base_colors = ['b','g','r']
    if len(unique_colors) <= len(base_colors):
        return create_color_dict(unique_colors,base_colors)
    if len(unique_colors) <= len(mcolors.BASE_COLORS):
        return create_color_dict(unique_colors,list(mcolors.BASE_COLORS))
    
    if len(unique_colors) <= len(mcolors.TABLEAU_COLORS):
        return create_color_dict(unique_colors,list(mcolors.TABLEAU_COLORS))
    
    return create_color_dict(unique_colors,list(mcolors.CSS4_COLORS))
           

def plot_scatter(x_values, y_values, color_values, x_label, y_label, color_dict, label_dict): 
    fig, ax = plt.subplots()
    for col in color_dict.keys(): 
        ix = np.where(color_values == col)
        ax.scatter(x_values[ix], y_values[ix], c = color_dict[col], label = label_dict[col], s = 10)
    ax.legend()
    plt.show()

def plot_params(df, x_label, y_label, color_label):
    le = LabelEncoder()
    int_values = le.fit_transform(df[color_label])
    unique_colors = list(set(int_values))
    color_dict = create_color_mapping(unique_colors)
    label_dict = dict()
    for color in unique_colors:
        label_dict[color] = le.inverse_transform([color])[0]
    return df[x_label].values, df[y_label].values,int_values, x_label, y_label,color_dict, label_dict
    
    

def plot_df(df, x_label, y_label, color_label):
    x_values, y_values, color_values, x_label, y_label, color_dict, label_dict = plot_params(df, x_label, y_label, color_label)
    plot_scatter(x_values, y_values, color_values, x_label, y_label, color_dict, label_dict)

def plot_2sides(df, x_label, y_label, color_label_1, color_label_2):
    x_values, y_values, color_values, x_label, y_label, color_dict, label_dict = plot_params(df, x_label, y_label, color_label_1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for col in color_dict.keys(): 
        ix = np.where(color_values == col)
        ax1.scatter(x_values[ix], y_values[ix], c = color_dict[col], label = label_dict[col], s = 10)
    ax1.legend()
    x_values, y_values, color_values, x_label, y_label, color_dict, label_dict = plot_params(df, x_label, y_label, color_label_2)
    for col in color_dict.keys(): 
        ix = np.where(color_values == col)
        ax2.scatter(x_values[ix], y_values[ix], c = color_dict[col], label = label_dict[col], s = 10)
    ax2.legend()
    plt.show()
    
    
