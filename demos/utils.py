import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn import preprocessing as skp 
from sklearn import decomposition as skd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


import pandas as pd


def plot_data(x_values, y_values):
    plt.figure(figsize=(6, 2))    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(x_values, y_values,c='b', marker=".") 

def plot_bar(x_data, y_data, xlabel, ylabel):
    plt.figure(figsize=(6, 2))    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.bar(x_data, y_data)
    plt.show()

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

def df_2_dict(df,key_column, value_column):
    return df[[key_column,value_column]].set_index(key_column)[value_column].to_dict()

def read_as_dict(filename, key_column, value_column):
    df = pd.read_csv(filename)
    return df_2_dict(df, key_column, value_column)

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
    le = skp.LabelEncoder()
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
    
def plot_multi_bar(df, selector_col, x_col, marker_col, figsize):
    all_cols = list(df.columns)
    plot_features = [x for x in all_cols if x not in [selector_col, x_col, marker_col]]
    num_features = len(plot_features)
    upper_limits = [df[x].max() for x in plot_features]
    lower_limits = [df[x].min() for x in plot_features]
    x_min = df[x_col].min()
    x_max = df[x_col].max()
    selectors = list(df[selector_col].unique())
    num_selectors = len(selectors)
    label_dict = df_2_dict(df[[selector_col,marker_col]].drop_duplicates(), selector_col, marker_col)
    fig, axs = plt.subplots(num_selectors, num_features, figsize=figsize)
    for index_1, value in enumerate(selectors): 
        for index_2, feature in enumerate(plot_features):
            sub_df = df[df[selector_col]==value]
            x_points = sub_df[x_col].tolist()
            y_points = sub_df[feature].tolist()
            this_plot = axs[index_1, index_2]
            this_plot.bar(x_points, y_points)
            y_label = f'{value}({label_dict.get(value,"Unknown")})'
            if index_2 == 0:
                this_plot.set(ylabel=y_label)
            else: 
                this_plot.set(ylabel='') 
            if index_1 == 0:
                this_plot.set_title(feature)
            else:
                this_plot.set_title('')
            this_plot.set_ylim([0, upper_limits[index_2]])
            this_plot.set_xlim([x_min, x_max])
            #this_plot.label_outer()
    plt.show()    

def determine_feature_importance(X, Y,feature_names):
    myfunctions = [('PreProcess', skp.MinMaxScaler()), ('Classifier', RandomForestClassifier() )]
    pipeline = Pipeline(myfunctions)
    pipeline.fit(X, Y)
    classifier = pipeline['Classifier']
    importances = classifier.feature_importances_
    this_list = [(x, importances[i]) for i,x in enumerate(feature_names)]
    answer = sorted(this_list, key=lambda x: x[1],reverse=True)
    return answer

def add_column(df, map_dict, field_name, orig_column, default_value):
    def get_new_value(x):
        return map_dict.get(x,default_value)
    df[field_name] = df[orig_column].apply(get_new_value)
    return df

def is_iterable(var):
    try:
        iter(var)
        return True
    except TypeError:
        return False

def filter_df(df, column, value):
    if isinstance(value, (int, float, str, bool)):
        df = df[df[column]!=value]
        return df
    elif is_iterable(value):
        for item in value:
            df = df[df[column]!=item]
        return df
    else:
        df = df[df[column]!=value]
        return df

def plot_pca_map(df, features_column, color_column):
    X = df[features_column].to_numpy()
    Y = df[color_column].to_list()
    scaler = skp.MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    pca = skd.PCA(n_components=2)
    pca_X = pca.fit_transform(scaled_X)
    pca_df = pd.DataFrame({'PCA_1':pca_X[:,0], 'PCA_2': pca_X[:,1], color_column:Y})
    plot_df(pca_df, 'PCA_1', 'PCA_2', color_column)
    