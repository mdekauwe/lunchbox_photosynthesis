#!/usr/bin/env python

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone
import os
import glob

def main(fname, lunchbox_width_cm, lunchbox_height_cm, lunchbox_length_cm,
         temp_k=295.15):

    st_autorefresh(interval=10_000, limit=None, key="co2_autorefresh")

    lunchbox_volume = calc_volume_litres(lunchbox_width_cm, lunchbox_height_cm,
                                         lunchbox_length_cm)

    df = load_and_process_data(fname, lunchbox_volume)

    if df.empty:
        st.info("Waiting for CO₂ data or unable to load data.")
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['time'], y=df['co2'], mode='lines+markers',
            name='CO₂ (ppm)',
            yaxis='y1',
            line=dict(color='green')
        ))

        fig.add_trace(go.Scatter(
            x=df['time'], y=df['anet'], mode='lines+markers',
            name='Anet (μmol s⁻¹)',
            yaxis='y2'
        ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis=dict(
                title="CO₂ (ppm)",
                side='left',
                showgrid=False,
            ),
            yaxis2=dict(
                title="Net Assimilation Rate (μmol s⁻¹)",
                side='right',
                showgrid=False,
                overlaying='y'
            ),
            legend=dict(
                y=0,
                x=1,
                xanchor='right',
                yanchor='bottom',
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12),
            ),
            margin=dict(r=150)
        )

        st.plotly_chart(fig, use_container_width=True)

def calc_volume_litres(width_cm, height_cm, length_cm):
    volume_cm3 = width_cm * height_cm * length_cm

    return volume_cm3 / 1000

def calc_anet(delta_ppm_s, lunchbox_volume_litres, temp_k=295.15):
    pressure = 101325.0  # Pa
    rgas = 8.314  # J/(K mol)
    volume_m3 = lunchbox_volume_litres / 1000.0  # litres to m3
    an_leaf = (delta_ppm_s * pressure * volume_m3) / (rgas * temp_k)

    return an_leaf  # umol leaf s-1

def load_and_process_data(fname, lunchbox_volume_l):
    if not os.path.exists(fname):
        st.warning("CSV file not found.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(fname, comment='#')
        df = df.rename(columns={
            'Timestamp UTC [Unix]': 'timestamp',
            'Timestamp Local [yyyy-MM-dd hh:mm:ss]': 'time',
            # The device port column may vary, so find the third col:
        })

        # Find the CO2 column dynamically (third column)
        co2_col = df.columns[2]
        df = df.rename(columns={co2_col: 'co2'})

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        # Calculate delta ppm / s
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'co2'])
        df['delta_seconds'] = df['timestamp'].diff()
        df['delta_ppm'] = df['co2'].diff()
        df['delta_ppm_s'] = df['delta_ppm'] / df['delta_seconds']

        # Calculate anet (NaN for first row)
        df['anet'] = df['delta_ppm_s'].apply(lambda x: calc_anet(x,
                                lunchbox_volume_l) if pd.notnull(x) else None)

        return df

    except Exception as e:
        st.error(f"Failed to read or process CSV file: {e}")
        return pd.DataFrame()

def find_latest_csv(desktop_path, prefix="PAS_CO2_datalog_"):
    pattern = os.path.join(desktop_path, prefix + "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort files by modification time descending and pick the newest
    files.sort(key=os.path.getmtime, reverse=True)

    return files[0]

if __name__ == "__main__":

    #fname = "/Users/xj21307/Desktop/PAS_CO2_datalog_20250719-183601.csv"
    fname = find_latest_csv(os.path.expanduser("~/Desktop"))
    lunchbox_width_cm = 17.5
    lunchbox_height_cm = 5
    lunchbox_length_cm = 12
    temp_k = 295.15  # Default temperature in Kelvin

    if fname is None:
        print("No CSV file starting with 'PAS_CO2_datalog_' found on Desktop")
    else:
        print(fname)
        main(fname, lunchbox_width_cm, lunchbox_height_cm, lunchbox_length_cm,
             temp_k)
