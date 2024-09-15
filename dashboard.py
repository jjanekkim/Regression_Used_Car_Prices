import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

df = pd.read_csv(r"C:\Users\nene0\Desktop\Projects\kaggle_Used_Car_Regression\cleaned_train.csv")

def groupby_data(data, group, column, agg_function):
    if agg_function == 'count':
        grouped_data = data.groupby(group)[column].count().sort_values(ascending=False).reset_index()
        return grouped_data
    elif agg_function == 'sum':
        grouped_data = data.groupby(group)[column].sum().sort_values(ascending=False).reset_index()
        return grouped_data
    elif agg_function == 'mean':
        grouped_data = data.groupby(group)[column].mean().sort_values(ascending=False).reset_index()
        return grouped_data
    elif agg_function == 'median':
        grouped_data = data.groupby(group)[column].median().sort_values(ascending=False).reset_index()
        return grouped_data

st.title("Used Car Regression Dashboard")

brands, vehicle_info, eng_trans, hist, rel, corr = st.tabs(["Top Brands", "Vehicle Info", "Engine & Transmission", "Histograms", "Relational Plots", "Correlation"])

with brands:

    fig = px.bar(groupby_data(df, 'brand', 'model', 'count'),
                x='brand',
                y='model',
                labels={'brand':'Brands', 'model':'Number of Cars'},
                text_auto=True,
                color='model',
                color_continuous_scale='haline_r')
    fig.update_layout(title='Number of Used Cars by Brands',
                      title_x=0.4)

    st.plotly_chart(fig)

with vehicle_info:

    fig = px.bar(groupby_data(df, 'fuel_type', 'model', 'count'),
             y='fuel_type',
             x='model',
             labels={'fuel_type':'Fuel Types', 'model':'Number of Cars'}, 
             text_auto=True,
             color='model',
             color_continuous_scale='haline_r')
    fig.update_layout(title='Number of Used Cars by Fuel Types',
                      title_x=0.4,
                      yaxis={'categoryorder': 'total ascending'})
    
    st.plotly_chart(fig)

    col1, col2 = st.columns([0.55,0.45])

    with col1:
        fig = px.histogram(df,
                        x='accident',
                        width=1000,
                        height=500,
                        text_auto=True,
                        labels={'accident':'Accident'},
                        color='accident',
                        color_discrete_sequence=['#003566', '#780000', '#f4a259'])
        fig.update_layout(title='How many cars had an accident?',
                        title_x=0.4,
                        showlegend=False)
        fig.update_yaxes(title_text='Number of Cars')
        fig.update_traces(textposition='outside')

        st.plotly_chart(fig)
    
    with col2:
        fig = px.histogram(df,
                        x='clean_title',
                        width=800,
                        height=500,
                        text_auto=True,
                        labels={'clean_title':'Title'},
                        color='clean_title',
                        color_discrete_sequence=['#003566', '#780000'])
        fig.update_layout(title='How many cars have a clean title?',
                        title_x=0.4,
                        showlegend=False)
        fig.update_yaxes(title_text='Number of Cars')
        fig.update_traces(textposition='outside')

        st.plotly_chart(fig)

with eng_trans:

    fig = px.bar(groupby_data(df, 'transmission', 'model', 'count'),
                y='transmission',
                x='model',
                labels={'transmission':'Transmission Types', 'model':'Number of Cars'}, 
                text_auto=True,
                color='model',
                color_continuous_scale='haline_r')
    fig.update_layout(title='Number of Used Cars by Transmission',
                    title_x=0.4,
                    yaxis={'categoryorder': 'total ascending'})
    
    st.plotly_chart(fig)

with hist:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(df,
                    x='age',
                    title="Histogram of Used Car's Age",
                    labels={'age':'Age'},
                    histnorm='probability density',
                    opacity=0.8,
                    color_discrete_sequence=['#168aad'],
                    width=800,
                    height=400)
        fig.update_layout(title_x=0.4)
        fig.update_yaxes(title_text='Probability Density')

        st.plotly_chart(fig)

    with col2:

        fig = px.histogram(df,
                    x='milage',
                    title="Histogram of Milage",
                    labels={'milage':'Milage'},
                    histnorm='probability density',
                    opacity=0.8,
                    color_discrete_sequence=['#f4a259'],
                    width=800,
                    height=400)
        fig.update_layout(title_x=0.4)
        fig.update_yaxes(title_text='Probability Density')

        st.plotly_chart(fig)

with corr:

    fig = px.imshow(df.corr(numeric_only=True),
                width=650,
                height=650,
                text_auto=True,
                color_continuous_scale='RdBu')
    fig.update_traces(texttemplate='%{z:.2f}')
    fig.update_layout(title='Correlation Matrix',
                      title_x=0.38)
    
    st.plotly_chart(fig)