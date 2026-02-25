import sys
import os
from bs4 import BeautifulSoup
import csv
import re

def extract_cable_data(html_file, output_csv):
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except Exception as e:
        print(f"Error reading file {html_file}: {e}")
        return

    tables = soup.find_all('table')
    
    # Locate the main cable list table
    target_table = None
    for table in tables:
        text = table.get_text()
        if "CABLE" in text and "NO." in text and "LEAVING POINT" in text:
            target_table = table
            break
    
    if not target_table:
        print("Could not find the cable list table.")
        # Fallback: Try the largest table
        if tables:
            target_table = max(tables, key=lambda t: len(str(t)))
            print("Using the largest table instead.")
        else:
            print("No tables found.")
            return

    rows_data = []
    
    # Process rows
    # The structure is complex with rowspan/colspan. 
    # For a simple extraction, we can try to flatten rows or just grabbing text from each tr.
    # A more robust approach for this specific layout:
    # Identify the header rows (skip them or extract them specially)
    # Identify data rows.
    
    # Let's try to extract all rows and let the user filter/clean for now, 
    # as 100% perfect parsing of a complex nested HTML table without visual rendering is hard.
    # However, we can try to be smart about columns.
    
    for tr in target_table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        row = []
        for cell in cells:
            # Join all text within the cell, stripping whitespace
            text = ' '.join(cell.stripped_strings)
            row.append(text)
        
        # specific cleanup for this file type if needed
        # Filter out empty rows if they have no content
        if any(row):
            rows_data.append(row)

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(rows_data)
        print(f"Successfully wrote {len(rows_data)} rows to {output_csv}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    input_file = r"C:\new_atlas\02(3)_CABLE LIST(VACUUM SYSTEM)_151-E8810-411-0(A4).html"
    output_file = r"C:\new_atlas\extracted_cable_list.csv"
    
    print(f"Extracting from {input_file}...")
    extract_cable_data(input_file, output_file)
