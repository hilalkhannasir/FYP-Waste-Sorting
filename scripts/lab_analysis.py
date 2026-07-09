import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from pathlib import Path


def get_image_lab_stats(image_path):
    img = io.imread(image_path)

    lab_img = color.rgb2lab(img)

    L = lab_img[:, :, 0]
    a = lab_img[:, :, 1]
    b = lab_img[:, :, 2]

    stats = {
        'L_min': np.min(L),
        'L_max': np.max(L),
        'L_mean': np.mean(L),
        'L_std': np.std(L),

        'a_min': np.min(a),
        'a_max': np.max(a),
        'a_mean': np.mean(a),
        'a_std': np.std(a),

        'b_min': np.min(b),
        'b_max': np.max(b),
        'b_mean': np.mean(b),
        'b_std': np.std(b),

    }


    return stats


def plot_class_boxplots(class_stats_dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    channel_data = {'L': [], 'a': [], 'b': []}
    class_names = []

    for class_name, channel_stats_df in class_stats_dict.items():
        class_names.append(class_name)

        L_mean = channel_stats_df[channel_stats_df['Channel'] == 'L']['Mean (all images)'].values[0]
        L_std = channel_stats_df[channel_stats_df['Channel'] == 'L']['Std (all images)'].values[0]
        L_min = channel_stats_df[channel_stats_df['Channel'] == 'L']['Min (all images)'].values[0]
        L_max = channel_stats_df[channel_stats_df['Channel'] == 'L']['Max (all images)'].values[0]

        a_mean = channel_stats_df[channel_stats_df['Channel'] == 'a']['Mean (all images)'].values[0]
        a_std = channel_stats_df[channel_stats_df['Channel'] == 'a']['Std (all images)'].values[0]
        a_min = channel_stats_df[channel_stats_df['Channel'] == 'a']['Min (all images)'].values[0]
        a_max = channel_stats_df[channel_stats_df['Channel'] == 'a']['Max (all images)'].values[0]

        b_mean = channel_stats_df[channel_stats_df['Channel'] == 'b']['Mean (all images)'].values[0]
        b_std = channel_stats_df[channel_stats_df['Channel'] == 'b']['Std (all images)'].values[0]
        b_min = channel_stats_df[channel_stats_df['Channel'] == 'b']['Min (all images)'].values[0]
        b_max = channel_stats_df[channel_stats_df['Channel'] == 'b']['Max (all images)'].values[0]

        L_data = [L_min, L_mean - L_std, L_mean, L_mean + L_std, L_max]
        a_data = [a_min, a_mean - a_std, a_mean, a_mean + a_std, a_max]
        b_data = [b_min, b_mean - b_std, b_mean, b_mean + b_std, b_max]

        channel_data['L'].append(L_data)
        channel_data['a'].append(a_data)
        channel_data['b'].append(b_data)

    channels = [
        ('L', 'Lightness (L)', 0, 100, axes[0]),
        ('a', 'Green-Red (a)', -128, 127, axes[1]),
        ('b', 'Blue-Yellow (b)', -128, 127, axes[2])
    ]

    for channel, title, ymin, ymax, ax in channels:
        bp = ax.bxp([
            {
                'med': data[2],
                'q1': data[1],
                'q3': data[3],
                'whislo': data[0],
                'whishi': data[4],
                'fliers':[],
                'label': class_names[i]
            }
            for i, data in enumerate(channel_data[channel])
        ], patch_artist=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{channel} Channel Distribution by Class', fontsize=14)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

        if len(class_names) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("class_comparison_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_folder(folder_path, output_file):
    folder_path = Path(folder_path)
    image_files = []

    image_files.extend(folder_path.glob('*.jpg'))
    image_files = list(set(image_files))

    print(f"Found {len(image_files)} images to process...")

    all_stats = []

    for idx, img_path in enumerate(image_files, 1):

        stats = get_image_lab_stats(img_path)

        stats['image_name'] = img_path.name
        all_stats.append(stats)


    df = pd.DataFrame(all_stats)

    column_order = [
        'image_name', 'L_min', 'L_max', 'L_mean', 'L_std',
        'a_min', 'a_max', 'a_mean', 'a_std',
        'b_min', 'b_max', 'b_mean', 'b_std'
    ]

    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(1)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='LAB Statistics', index=False)

        channel_stats = {
            'Channel': ['L', 'a', 'b'],
            'Mean (all images)': [
                df['L_mean'].mean(),
                df['a_mean'].mean(),
                df['b_mean'].mean()
            ],
            'Std (all images)': [
                df['L_std'].mean(),
                df['a_std'].mean(),
                df['b_std'].mean()
            ],
            'Min (all images)': [
                df['L_min'].min(),
                df['a_min'].min(),
                df['b_min'].min()
            ],
            'Max (all images)': [
                df['L_max'].max(),
                df['a_max'].max(),
                df['b_max'].max()
            ]
        }
        channel_stats_df = pd.DataFrame(channel_stats)
        numeric_columns = channel_stats_df.select_dtypes(include=[np.number]).columns
        channel_stats_df[numeric_columns] = channel_stats_df[numeric_columns].round(1)
        channel_stats_df.to_excel(writer, sheet_name='Channel Statistics', index=False)

    print(f"   Processed: {len(all_stats)} images")
    return channel_stats_df


CLASSES = ["cardboard","cloth","egg-shell","organic","plastic","plastic-bag","tea-waste"]
channel_stats = {}
for cls in CLASSES:
    channel_stats[cls] = process_folder(f'crops/{cls}',f'{cls}_lab_analysis.xlsx')

plot_class_boxplots(channel_stats)