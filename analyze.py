import pandas as pd
import re
import numpy as np
from pathlib import Path

def parse_performance_data(text_data):
    """
    Parse performance testing data from text format
    Returns a structured dictionary with organized results
    """
    # Split the data into sections
    sections = re.split(r'={20,}', text_data)
    
    parsed_data = []
    i = 0
    
    while i < len(sections):
        section = sections[i].strip()
        if not section:
            i += 1
            continue
            
        # Check if this is a header line
        if 'with' in section and '_LOAD' in section:
            # Extract service name and load type
            header_match = re.search(r'(\w+(?:-\w+)*)\s+with\s+(\w+_LOAD)', section)
            if not header_match:
                i += 1
                continue
                
            current_service = header_match.group(1)
            current_load = header_match.group(2)
            
            # Look for the next section which should contain the data
            if i + 1 < len(sections):
                data_section = sections[i + 1].strip()
                lines = data_section.split('\n')
                
                # Initialize statistics
                stats = {
                    'service': current_service,
                    'load_type': current_load,
                    'mean': None,
                    'median': None,
                    'p90': None,
                    'p95': None,
                    'p99': None,
                    'all_times': []
                }
                
                # Parse the data section
                numeric_values = []
                all_times_values = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if it's the all times array
                    if line.startswith('[') and line.endswith(']'):
                        array_str = line[1:-1]  # Remove brackets
                        try:
                            all_times_values = [float(x.strip()) for x in array_str.split(',')]
                        except ValueError:
                            continue
                    # Skip the "all times note" line
                    elif 'all times note' in line:
                        continue
                    else:
                        # Try to parse as numeric value
                        try:
                            value = float(line)
                            numeric_values.append(value)
                        except ValueError:
                            continue
                
                # Based on your original format description, the 5 numeric values should be:
                # mean, median, p90, p95, p99
                if len(numeric_values) >= 5:
                    stats['mean'] = numeric_values[0]
                    stats['median'] = numeric_values[1]
                    stats['p90'] = numeric_values[2]
                    stats['p95'] = numeric_values[3]
                    stats['p99'] = numeric_values[4]
                
                stats['all_times'] = all_times_values
                
                # Store the parsed data if we have any useful information
                if (any(stats[key] is not None for key in ['mean', 'median', 'p90', 'p95', 'p99']) 
                    or stats['all_times']):
                    parsed_data.append(stats)
                
                # Skip the data section since we just processed it
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return parsed_data

def create_summary_dataframe(parsed_data):
    """
    Create a summary DataFrame with statistics for each service/load combination
    """
    summary_rows = []
    
    for entry in parsed_data:
        service = entry['service']
        load_type = entry['load_type']
        all_times = entry['all_times']
        
        # Calculate additional statistics from raw data if available
        additional_stats = {}
        if all_times:
            additional_stats = {
                'Count': len(all_times),
                'Calculated_Min': min(all_times),
                'Calculated_Max': max(all_times),
                'Calculated_Mean': np.mean(all_times),
                'Calculated_Median': np.median(all_times),
                'Std_Dev': np.std(all_times)
            }
        
        summary_rows.append({
            'Service': service,
            'Load_Type': load_type,
            'Reported_Mean': entry['mean'],
            'Reported_Median': entry['median'],
            'Reported_P90': entry['p90'],
            'Reported_P95': entry['p95'],
            'Reported_P99': entry['p99'],
            **additional_stats
        })
    
    return pd.DataFrame(summary_rows)

def create_detailed_dataframe(parsed_data):
    """
    Create a detailed DataFrame with all individual timing measurements
    """
    detailed_rows = []
    
    for entry in parsed_data:
        service = entry['service']
        load_type = entry['load_type']
        all_times = entry['all_times']
        
        if all_times:
            for i, timing in enumerate(all_times):
                detailed_rows.append({
                    'Service': service,
                    'Load_Type': load_type,
                    'Measurement_Index': i + 1,
                    'Timing_Value': timing
                })
    
    return pd.DataFrame(detailed_rows)

def create_pivot_table(detailed_df):
    """
    Create pivot tables for each service showing statistics across load types
    Returns a dictionary of DataFrames, one for each service
    """
    if detailed_df.empty:
        return {}
    
    # Get unique services
    services = detailed_df['Service'].unique()
    pivot_tables = {}
    
    for service in services:
        # Filter data for this service
        service_data = detailed_df[detailed_df['Service'] == service]
        
        # Calculate statistics for each load type
        stats_data = []
        load_types = service_data['Load_Type'].unique()
        
        # Initialize the statistics dictionary
        stats_dict = {
            'Statistic': ['Min', 'Max', 'Mean', 'Median', 'P90', 'P95', 'P99', 'Std Dev']
        }
        
        # Calculate statistics for each load type
        for load_type in sorted(load_types):  # Sort to ensure consistent order
            load_data = service_data[service_data['Load_Type'] == load_type]['Timing_Value']
            
            if len(load_data) > 0:
                stats_dict[load_type] = [
                    load_data.min(),                    # Min
                    load_data.max(),                    # Max
                    load_data.mean(),                   # Mean
                    load_data.median(),                 # Median
                    load_data.quantile(0.90),          # P90
                    load_data.quantile(0.95),          # P95
                    load_data.quantile(0.99),          # P99
                    load_data.std()                     # Std Dev
                ]
            else:
                stats_dict[load_type] = [0] * 8  # Fill with zeros if no data
        
        # Create DataFrame for this service
        pivot_df = pd.DataFrame(stats_dict)
        pivot_df.set_index('Statistic', inplace=True)
        pivot_tables[service] = pivot_df
    
    return pivot_tables

def debug_parsing(text_data):
    """
    Debug function to show what's being parsed
    """
    print("DEBUG: Starting parsing debug...")
    sections = re.split(r'={20,}', text_data)
    print(f"DEBUG: Found {len(sections)} sections after splitting")
    
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            print(f"DEBUG: Section {i} is empty")
            continue
        
        print(f"DEBUG: Section {i}:")
        print(f"First 100 chars: {section[:100]}")
        
        # Check if this is a header line
        if 'with' in section and '_LOAD' in section:
            header_match = re.search(r'(\w+(?:-\w+)*)\s+with\s+(\w+_LOAD)', section)
            if header_match:
                print(f"DEBUG: Found service '{header_match.group(1)}' with load '{header_match.group(2)}'")
            else:
                print("DEBUG: Header pattern not matched")
        
        lines = section.split('\n')
        print(f"DEBUG: Section has {len(lines)} lines")
        for j, line in enumerate(lines[:5]):  # Show first 5 lines
            print(f"DEBUG: Line {j}: '{line.strip()}'")
        print("---")

def main():
    # Read data
    with open('run-all-out.txt', 'r') as f:
        text_data = f.read()
    
    print("Parsing performance data...")
    parsed_data = parse_performance_data(text_data)
    print(f"Found data for {len(parsed_data)} service/load combinations")
    
    if len(parsed_data) == 0:
        print("ERROR: No data was parsed!")
        debug_parsing(text_data)
        return
    
    # Create DataFrames and save to Excel
    summary_df = create_summary_dataframe(parsed_data)
    detailed_df = create_detailed_dataframe(parsed_data)
    pivot_tables = create_pivot_table(detailed_df)
    
    # Save to Excel
    output_file = 'performance_analysis.xlsx'
    print(f"Saving results to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        detailed_df.to_excel(writer, sheet_name='Detailed_Measurements', index=False)
        
        # Save all pivot tables on one sheet
        if pivot_tables:
            # Create a single sheet for all pivot tables
            workbook = writer.book
            pivot_sheet = workbook.create_sheet('Pivot_Tables')
            
            current_row = 1
            for service, pivot_df in pivot_tables.items():
                # Write service name as header
                pivot_sheet.cell(row=current_row, column=1, value=f"{service.upper()} SERVICE")
                current_row += 2
                
                # Write the pivot table
                # First write column headers
                pivot_sheet.cell(row=current_row, column=1, value="Statistic")
                for col_idx, col_name in enumerate(pivot_df.columns, start=2):
                    pivot_sheet.cell(row=current_row, column=col_idx, value=col_name)
                current_row += 1
                
                # Write data rows
                for row_idx, (stat_name, row_data) in enumerate(pivot_df.iterrows()):
                    pivot_sheet.cell(row=current_row, column=1, value=stat_name)
                    for col_idx, value in enumerate(row_data, start=2):
                        pivot_sheet.cell(row=current_row, column=col_idx, value=value)
                    current_row += 1
                
                # Add spacing between tables
                current_row += 2
                
                print(f"Added pivot table for service: {service}")
        
        # Raw data
        raw_data_rows = []
        for entry in parsed_data:
            raw_data_rows.append({
                'Service': entry['service'],
                'Load_Type': entry['load_type'],
                'Mean': entry['mean'],
                'Median': entry['median'],
                'P90': entry['p90'],
                'P95': entry['p95'],
                'P99': entry['p99'],
                'All_Times_Count': len(entry['all_times']),
                'All_Times': str(entry['all_times'])
            })
        
        raw_df = pd.DataFrame(raw_data_rows)
        raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)
    
    print(f"Excel file saved successfully: {output_file}")
    print(f"Summary DataFrame has {len(summary_df)} rows")
    print(f"Detailed DataFrame has {len(detailed_df)} rows")
    print(f"Created {len(pivot_tables)} pivot tables on single sheet")

if __name__ == "__main__":
    main()