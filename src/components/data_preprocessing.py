import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logging

# Ensure the 'plots' directory exists
os.makedirs('plots', exist_ok=True)

# Utility function to save plots
def save_plot(fig, filename):
    fig.savefig(f'plots/{filename}')
    plt.close(fig)

def plot_pie_chart(df, column, title, filename):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(x=df[column].value_counts(), labels=df[column].value_counts().index, autopct='%.02f%%')
    ax.set_title(title)
    save_plot(fig, filename)

def plot_point_plot(data, x_col, y_col, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.pointplot(data=data, x=x_col, y=y_col, ax=ax)
    ax.tick_params(axis='x', labelrotation = 90)#ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7) # plt.xticks(rotation=90,fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_plot(fig, filename)

def plot_count_plots(df, columns, hue, custom_palette, filename):
    cols = 3
    rows = 6#py(len(columns) + cols - 1) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    for ax, column in zip(axes, columns):
        sns.countplot(x=column, hue=hue, data=df, ax=ax, palette=custom_palette)
        ax.set_title(column)

    for ax in axes[len(columns):]:
        fig.delaxes(ax)

    plt.tight_layout()
    save_plot(fig, filename)

def plot_kde_and_hist(df, numerical_columns, custom_palette, filename):
    cols = 2
    rows = 2#(len(numerical_columns) + cols - 1) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(numerical_columns):
         # Plot KDE plot
        sns.kdeplot(data=df, x=col, hue='Churn', ax=axes[i], palette=custom_palette)
        axes[i].set_title(f'KDE Plot of {col} by Churn')
        axes[i].legend()
        axes[i].grid(True)

        # Plot histogram with KDE overlay
        sns.histplot(data=df, x=col, hue='Churn', ax=axes[i+cols], palette=custom_palette, kde=True)
        axes[i+cols].set_title(f'Distribution of {col} by Churn')
        axes[i+cols].legend()
        axes[i+cols].grid(True)
        # sns.kdeplot(data=df, x=col, hue='Churn', ax=axes[i], palette=custom_palette)
        # axes[i].set_title(f'KDE Plot of {col} by Churn')
        # axes[i].legend()
        # axes[i].grid(True)

    plt.tight_layout()
    save_plot(fig, filename)

def preprocessing(df):
    logging.info("Preprocessing & Visualization Started")
    
    # Pie plot for Churn
    plot_pie_chart(df, 'Churn', 'Churn', 'ChurnCircle.png')

    # Null values count point plot
    missing = (((df.isna().sum()) * 100) / df.shape[0]).reset_index()
    plot_point_plot(missing, 'index', 0, "Percentage of Missing Value", "Columns", "Percentage", 'Nullvalues.png')

    # Convert TotalCharges to numeric and recheck missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_total_charge = (((df.isnull().sum()) * 100) / df.shape[0]).reset_index()
    plot_point_plot(missing_total_charge, 'index', 0, "Percentage of Missing Value", "Columns", "Percentage", 'TotalCharges.png')

    # Drop null values
    df.dropna(how='any', inplace=True)

    # Group tenure and drop unnecessary columns
    labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    df['tenure_group'] = pd.cut(df['tenure'], bins=range(1, 80, 12), labels=labels, right=False)
    df.drop(columns=['customerID', 'tenure'], axis=1, inplace=True)

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()

    categorical_columns.append('SeniorCitizen')
    numerical_columns.remove('SeniorCitizen')

    # Count plot of each column
    custom_palette = {'Yes': 'red', 'No': 'green'}
    plot_count_plots(df, categorical_columns, 'Churn', custom_palette, 'count_of_each_column.png')

    # Hist plot of Churn
    plot_kde_and_hist(df, numerical_columns, custom_palette, 'churn_histograms_kde.png')

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=numerical_columns[0], y=numerical_columns[1], ax=ax)
    save_plot(fig, 'scatter_plot.png')

    # Additional count plots
    count_plots = [
        ('Partner', 'Dependents', 'Partner vs Dependents', 'partner_vs_dependents.png'),
        ('InternetService', 'PaymentMethod', 'InternetService vs PaymentMethod', 'InternetService_vs_PaymentMethod.png'),
        ('Partner', 'OnlineSecurity', 'Partner vs OnlineSecurity', 'Partner_vs_OnlineSecurity.png')
    ]

    for x, hue, title, filename in count_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=df[x], hue=df[hue], ax=ax)
        ax.set_title(title)
        save_plot(fig, filename)

    logging.info('preprocessing & visualization is done')
    

# Example usage:
# df = pd.read_csv('path_to_your_data.csv')
# preprocessing(df)

# if __name__ == '__main__':
#     data=pd.read_csv('./artifacts/data.csv')
#     preprocessing(data)