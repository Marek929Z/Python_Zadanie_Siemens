import pandas as pd
import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

# load configuration YAML file
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# convert numeric columns from string (object) to float
def convert_numeric(df):
    case_name = df['CaseName'] if 'CaseName' in df.columns else None
    df = df.drop(columns=['CaseName'], errors='ignore')

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace(',', '.')
            df[column] = pd.to_numeric(df[column], errors='coerce')

    if case_name is not None:
        df.insert(0, 'CaseName', case_name)

    return df

# modify dataset by changing values, introducing/removing NaN values, and adding or removing samples
def modify_dataset(df, config):
    rng = np.random.default_rng(config.get('random_seed', None))
    df_modified = df.copy()

    for column in df_modified.columns:
        if pd.api.types.is_numeric_dtype(df_modified[column]):
            if rng.random() < config['modification']['probability_change_values']:
                df_modified[column] *= rng.uniform(0.8, 1.2, size=len(df_modified))

            flip_mask = rng.random(len(df_modified)) < config['modification']['probability_missing']
            for i in range(len(df_modified)):
                if flip_mask[i]:
                    if pd.isna(df_modified.at[i, column]):
                        col_mean = df_modified[column].mean(skipna=True)
                        df_modified.at[i, column] = col_mean if not np.isnan(col_mean) else 0
                    else:
                        df_modified.at[i, column] = np.nan

    if rng.random() < config['modification']['probability_add_sample']:
        new_row = df.sample(1, random_state=config.get('random_seed', 42)).copy()

        if 'CaseName' in df.columns:
            existing_names = df['CaseName'].astype(str)
            numeric_parts = existing_names.str.extract(r'(\d+)$')[0].dropna().astype(int)
            new_index = numeric_parts.max() + 1 if not numeric_parts.empty else 1
            new_row['CaseName'] = f"Case {new_index:03d}"

        df_modified = pd.concat([df_modified, new_row], ignore_index=True)

    elif rng.random() < config['modification']['probability_add_sample'] and len(df_modified) > 0:
        drop_idx = rng.integers(len(df_modified))
        df_modified = df_modified.drop(index=drop_idx).reset_index(drop=True)

    return df_modified

# compute statistics for numeric columns
def compute_statistics(df):
    case_name = df['CaseName'] if 'CaseName' in df.columns else None
    df = df.drop(columns=['CaseName'], errors='ignore')

    stats = {
        'mean': df.mean(),
        'std': df.std(),
        'var': df.var(),
        'max': df.max(),
        'min': df.min(),
        'median': df.median(),
        'p5': df.quantile(0.05),
        'p95': df.quantile(0.95),
    }

    if case_name is not None:
        df.insert(0, 'CaseName', case_name)

    return pd.DataFrame(stats)

# compare values between datasets using thresholds from config file
def compare_values(gt_df, cmp_df, thresholds):
    differences = []
    merged = pd.merge(gt_df, cmp_df, on="CaseName", suffixes=('_gt', '_cmp'), how='outer', indicator=True)

    for _, row in merged.iterrows():
        case_name = row['CaseName']

        if row['_merge'] == 'left_only':
            differences.append({
                'Measurement': case_name,
                'Attribute': 'Missing in Comparison',
                'GT': 'Exists',
                'Modify': 'Missing',
                'Diff_%': 'N/A'
            })
        elif row['_merge'] == 'right_only':
            differences.append({
                'Measurement': case_name,
                'Attribute': 'New Sample in Comparison',
                'GT': 'Missing',
                'Modify': 'Exists',
                'Diff_%': 'N/A'
            })
        else:
            for column in gt_df.columns:
                if column == 'CaseName' or not pd.api.types.is_numeric_dtype(gt_df[column]):
                    continue

                gt_val = row.get(f'{column}_gt')
                cmp_val = row.get(f'{column}_cmp')

                if pd.isna(gt_val) or pd.isna(cmp_val):
                    continue

                threshold = thresholds.get('per_measurement', {}).get(column.strip(), thresholds.get("global", 0))
                diff_pct = abs(gt_val - cmp_val) / (abs(gt_val) if gt_val != 0 else 1) * 100

                if diff_pct > threshold:
                    differences.append({
                        'Measurement': case_name,
                        'Attribute': column,
                        'GT': gt_val,
                        'Modify': cmp_val,
                        'Diff_%': round(diff_pct, 2)
                    })

    return pd.DataFrame(differences)

# generate statistics, reports, visualizations and export PDF report
def generate_report(gt_df, cmp_df, config):
    gt_stats = compute_statistics(gt_df)
    cmp_stats = compute_statistics(cmp_df)
    differences = compare_values(gt_df, cmp_df, config['thresholds'])

    gt_stats.to_csv(config['data']['gt_statistics_path'], index=True)
    cmp_stats.to_csv(config['data']['cmp_statistics_path'], index=True)
    differences.to_csv(config['data']['report_path'], index=False)


    print("Report generated:")
    print(f"- GT Stats: {config['data']['gt_statistics_path']}")
    print(f"- Comparison Stats: {config['data']['cmp_statistics_path']}")
    print(f"- Differences: {config['data']['report_path']}")

    graphs_path = config['data']['graphs_path']
    os.makedirs(graphs_path, exist_ok=True)
    density_path = os.path.join(graphs_path, 'density')
    #boxplot_path = os.path.join(graphs_path, 'boxplots')
    density_path = os.path.join(graphs_path, config['data'].get('density_subfolder', 'density'))
    boxplot_path = os.path.join(graphs_path, config['data'].get('boxplot_subfolder', 'boxplots'))
    histogram_path = os.path.join(graphs_path,config['data'].get('histogram_subfolder','histograms'))

    os.makedirs(density_path, exist_ok=True)
    os.makedirs(boxplot_path, exist_ok=True)
    os.makedirs(histogram_path, exist_ok=True)

    pdf_path = config['data']['pdf_report_path']
    with PdfPages(pdf_path) as pdf:

        def render_dataframe_as_pdf_page(df, title, include_index_as_metric=True):
            df = df.copy()
            if include_index_as_metric and (df.index.name or df.index.name is None):
                df.insert(0, 'Metric', df.index)
                df.reset_index(drop=True, inplace=True)

            fig, ax = plt.subplots(figsize=(12, min(0.4 * len(df), 25)))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.round(2).values,
                             colLabels=df.columns,
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            plt.title(title, fontsize=14)
            pdf.savefig()
            plt.close()

        render_dataframe_as_pdf_page(gt_stats.transpose(), 'GT Dataset Statistics', include_index_as_metric=True)
        render_dataframe_as_pdf_page(cmp_stats.transpose(), 'Modified Dataset Statistics', include_index_as_metric=True)

        if not differences.empty:
            rows_per_page = 40
        total_pages = (len(differences) + rows_per_page - 1) // rows_per_page

        for i in range(total_pages):
            start = i * rows_per_page
            end = start + rows_per_page
            diff_page = differences.iloc[start:end]
            title = f"Differences (page {i+1}/{total_pages})"
            render_dataframe_as_pdf_page(diff_page, title, include_index_as_metric=False)

        for column in gt_df.columns:
            if column == 'CaseName' or not pd.api.types.is_numeric_dtype(gt_df[column]):
                continue
            
            # density graphs
            plt.figure(figsize=(10, 6))
            sns.kdeplot(gt_df[column], label='Ground Truth', fill=True)
            sns.kdeplot(cmp_df[column], label='Modified', fill=True)
            plt.title(f"Density Comparison of {column}")
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(os.path.join(density_path, f"{column}_density_comparison.png"))
            pdf.savefig()
            plt.close()

            # boxplots
            boxplot_data = pd.DataFrame({
                column: pd.concat([gt_df[column], cmp_df[column]]),
                'Dataset': ['GT'] * len(gt_df) + ['Modified'] * len(cmp_df)
            }).reset_index(drop=True)

            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Dataset', y=column, data=boxplot_data)
            plt.title(f'Boxplot Comparison for {column}')
            plt.savefig(os.path.join(boxplot_path, f"{column}_boxplot_comparison.png"))
            pdf.savefig()
            plt.close()

            # histograms
            plt.figure(figsize=(10, 6))
            plt.hist(gt_df[column].dropna(), bins=30, alpha=0.5, label='Ground Truth', edgecolor='black')
            plt.hist(cmp_df[column].dropna(), bins=30, alpha=0.5, label='Modified', edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(histogram_path, f"{column}_histogram_comparison.png"))
            pdf.savefig()
            plt.close()


    print(f"- PDF report generated: {pdf_path}")
    print(f"- Density plots: {density_path}")
    print(f"- Boxplots: {boxplot_path}")
    print(f"- Histograms: {histogram_path}")

# main function of program
def main(config_path, mode='all'):
    config = load_config(config_path)

    if mode in ['all', 'stats']:
        gt_df = pd.read_csv(config['data']['input_dataset_path'], sep=';')
        gt_df = convert_numeric(gt_df)
        cmp_df = modify_dataset(gt_df.copy(), config)
        gt_df.to_csv(config['data']['gt_dataset_path'], index=False)
        cmp_df.to_csv(config['data']['cmp_dataset_path'], index=False)
    else:
        gt_df = pd.read_csv(config['data']['gt_dataset_path'])
        cmp_df = pd.read_csv(config['data']['cmp_dataset_path'])

    if mode in ['all', 'compare']:
        generate_report(gt_df, cmp_df, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Comparison Tool")
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--mode', choices=['all', 'stats', 'compare'], default='all', help='Mode of operation')
    args = parser.parse_args()
    main(args.config, args.mode)