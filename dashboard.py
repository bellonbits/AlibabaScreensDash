import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to clean and preprocess data
def preprocess_data(df):
    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: x.strip())
    
    # Clean 'Price Range A (Ksh)' and 'Price Range B (Ksh)'
    df['Price Range A (Ksh)'] = df['Price Range A (Ksh)'].str.replace('Ksh', '').str.replace(',', '').astype(float)
    df['Price Range B (Ksh)'] = df['Price Range B (Ksh)'].str.replace('Ksh', '').str.replace(',', '').astype(float)
    
    # Clean 'Sold' column, removing non-numeric values
    df['Sold'] = df['Sold'].str.replace(',', '').str.extract('(\d+)').astype(float)
    
    # Convert 'Sold' to integer, handling NaN values
    df['Sold'] = df['Sold'].fillna(0).astype(int)
    
    # Convert 'Ratings (out of 5)' to numeric
    df['Ratings (out of 5)'] = pd.to_numeric(df['Ratings (out of 5)'], errors='coerce')
    
    return df

# Load and preprocess the data
@st.cache
def load_data():
    df = pd.read_csv('Alibaba.csv', encoding='latin1')
    return preprocess_data(df)

df = load_data()

# Streamlit Sidebar for Filters
st.sidebar.header("Filters")
product_type = st.sidebar.multiselect('Select Product Type', df['Type'].unique())
rating_range = st.sidebar.slider('Select Rating Range', min_value=0, max_value=5, value=(0, 5))
price_range = st.sidebar.slider('Select Price Range (Ksh)', min_value=float(df['Price Range A (Ksh)'].min()), 
                                 max_value=float(df['Price Range B (Ksh)'].max()), value=(float(df['Price Range A (Ksh)'].min()), float(df['Price Range B (Ksh)'].max())))

filtered_df = df[(df['Type'].isin(product_type)) & 
                 (df['Ratings (out of 5)'].between(rating_range[0], rating_range[1])) &
                 (df['Price Range A (Ksh)'] >= price_range[0]) & 
                 (df['Price Range B (Ksh)'] <= price_range[1])]

# Analysis and Visualizations
st.title("Product Analysis Dashboard")

# Summary Statistics
st.header("Summary Statistics")
st.write(f"Total Products: {filtered_df.shape[0]}")
st.write(f"Average Rating: {filtered_df['Ratings (out of 5)'].mean():.2f}")
st.write(f"Total Units Sold: {filtered_df['Sold'].sum()}")
st.write(f"Average Price Range A: {filtered_df['Price Range A (Ksh)'].mean():.2f} Ksh")
st.write(f"Average Price Range B: {filtered_df['Price Range B (Ksh)'].mean():.2f} Ksh")

# Sales Distribution by Product Type
st.header("Sales Distribution by Product Type")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Type', y='Sold', data=filtered_df, ax=ax)
plt.title('Sales Distribution by Product Type')
plt.xticks(rotation=45)
st.pyplot(fig)

# Ratings Distribution
st.header("Ratings Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['Ratings (out of 5)'], bins=10, kde=True, ax=ax)
plt.title('Ratings Distribution')
st.pyplot(fig)

# Price Range Distribution by Product Type
st.header("Price Range Distribution by Product Type")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Type', y='Price Range B (Ksh)', data=filtered_df, ax=ax)
plt.title('Price Range Distribution by Product Type')
plt.xticks(rotation=45)
st.pyplot(fig)

# Top Products by Units Sold
st.header("Top Products by Units Sold")
top_products = filtered_df.groupby('Product Title')['Sold'].sum().reset_index()
top_products = top_products.sort_values(by='Sold', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='Sold', y='Product Title', data=top_products, ax=ax)
plt.title('Top Products by Units Sold')
st.pyplot(fig)

# Average Rating by Product Type
st.header("Average Rating by Product Type")
avg_rating_by_type = filtered_df.groupby('Type')['Ratings (out of 5)'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Type', y='Ratings (out of 5)', data=avg_rating_by_type, ax=ax)
plt.title('Average Rating by Product Type')
plt.xticks(rotation=45)
st.pyplot(fig)

# Product with Highest Rating
st.header("Product with Highest Rating")
highest_rated_product = filtered_df[filtered_df['Ratings (out of 5)'] == filtered_df['Ratings (out of 5)'].max()]
st.write(highest_rated_product[['Product Title', 'Ratings (out of 5)', 'Sold']])

# Product with Minimum and Maximum Price for LEDs
st.header("Price Range for LEDs")
leds_df = filtered_df[filtered_df['Type'].str.upper() == 'LED']
min_price_led = leds_df['Price Range A (Ksh)'].min()
max_price_led = leds_df['Price Range B (Ksh)'].max()
st.write(f"Minimum Price for LEDs: {min_price_led:.2f} Ksh")
st.write(f"Maximum Price for LEDs: {max_price_led:.2f} Ksh")

