import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
# --- Data Preparation ---

# Data 1: Token Counts
df1_data = {1: {"num_pred_tokens": [68, 25, 26, 28, 10, 26, 19, 23, 21, 39, 27, 41, 20, 25, 27, 42, 24, 24, 20, 42, 39, 16, 20, 24, 38, 26, 44, 25, 31, 17, 29, 13, 15, 15, 21, 48, 45, 52, 17, 45, 36, 32, 20, 59, 29, 28, 21, 8, 24, 23, 21, 43, 35, 43, 45, 37, 47, 24, 20, 21, 35, 35, 26, 29, 17, 22, 37, 14, 27, 27, 38, 27, 7, 16, 23, 10, 38, 38, 27, 28, 53, 15, 27, 24, 49, 23, 42, 26, 36, 42, 27, 34, 30, 27, 44, 16, 26, 39, 28, 62], "num_ref_tokens": [47, 29, 34, 31, 11, 24, 20, 25, 20, 25, 26, 35, 19, 28, 27, 40, 22, 27, 24, 26, 26, 13, 35, 31, 35, 34, 45, 22, 34, 17, 31, 15, 15, 15, 20, 46, 59, 60, 19, 35, 37, 23, 19, 66, 36, 30, 21, 7, 24, 31, 34, 44, 33, 40, 46, 34, 48, 28, 28, 23, 33, 61, 28, 31, 23, 19, 41, 12, 32, 28, 54, 30, 6, 20, 25, 10, 48, 25, 39, 25, 48, 13, 30, 30, 51, 27, 39, 20, 35, 38, 26, 25, 76, 26, 24, 16, 25, 35, 30, 47]}, 2: {"num_pred_tokens": [60, 69, 34, 39, 65, 45, 52, 67, 36, 39, 32, 47, 43, 75, 43, 39, 36, 82, 149, 69, 59, 129, 73, 33, 109, 125, 71, 54, 368, 46, 56, 76, 39, 45, 58, 73, 21, 27, 66, 31, 53, 57, 48, 72, 89, 76, 63, 114, 39, 66, 46, 61, 54, 35, 57, 101, 57, 53, 34, 80, 38, 26, 71, 30, 38, 22, 55, 43, 41, 28, 70, 28, 35, 42, 146, 52, 77, 34, 70, 102, 61, 38, 36, 72, 49, 57, 33, 50, 40, 37, 34, 67, 53, 37, 37, 63, 21, 36, 28, 53], "num_ref_tokens": [76, 65, 35, 45, 45, 61, 47, 67, 49, 50, 39, 66, 69, 67, 51, 46, 30, 66, 119, 54, 60, 85, 66, 28, 55, 78, 73, 80, 76, 51, 94, 59, 42, 53, 60, 84, 26, 35, 73, 64, 61, 60, 78, 59, 73, 51, 102, 40, 60, 77, 52, 39, 84, 58, 55, 81, 89, 91, 45, 119, 32, 49, 102, 46, 64, 42, 74, 55, 157, 33, 95, 66, 81, 54, 53, 59, 73, 38, 73, 59, 57, 36, 49, 98, 53, 69, 82, 66, 76, 46, 33, 41, 63, 34, 44, 51, 40, 43, 59, 48]}, 3: {"num_pred_tokens": [107, 41, 48, 68, 46, 56, 58, 21, 59, 52, 60, 74, 270, 73, 67, 102, 69, 86, 65, 55, 348, 56, 32, 99, 37, 68, 48, 77, 372, 124, 96, 58, 78, 67, 83, 98, 85, 201, 52, 60, 72, 53, 41, 87, 56, 108, 31, 34, 39, 343, 74, 68, 98, 79, 40, 68, 56, 72, 52, 23, 58, 68, 65, 54, 37, 59, 69, 45, 82, 72, 71, 62, 58, 59, 41, 52, 51, 82, 42, 82, 38, 54, 36, 74, 81, 83, 46, 81, 82, 78, 82, 45, 82, 82, 75, 60, 65, 35, 89, 30], "num_ref_tokens": [110, 66, 65, 86, 74, 89, 76, 79, 114, 73, 61, 81, 138, 95, 121, 58, 89, 117, 128, 79, 122, 73, 85, 112, 51, 83, 112, 73, 117, 93, 127, 66, 90, 99, 75, 106, 79, 146, 113, 142, 57, 126, 98, 54, 95, 191, 86, 108, 104, 84, 77, 93, 107, 82, 74, 109, 87, 117, 121, 67, 61, 76, 51, 78, 54, 88, 71, 120, 122, 101, 59, 117, 108, 83, 71, 87, 87, 106, 79, 130, 37, 73, 81, 87, 60, 127, 110, 143, 107, 71, 88, 96, 122, 115, 111, 57, 84, 93, 104, 111]}, 4: {"num_pred_tokens": [169, 408, 318, 74, 510, 353, 353, 143, 22, 134, 512, 446, 368, 143, 125, 119, 93, 125, 315, 385, 260, 231, 84, 119, 138, 200, 343, 440, 345, 398, 162, 134, 31, 382, 362, 356, 68, 129, 204, 187, 337, 175, 173, 72, 151, 126, 132, 120, 336, 361, 100, 322, 458, 332, 363, 197, 359, 359, 121, 16, 167, 245, 160, 82, 354, 425, 108, 27, 130, 200, 152, 105, 294, 345, 116, 141, 138, 306, 334, 144, 85, 203, 142, 151, 173, 106, 183, 95, 271, 152, 201, 179, 328, 400, 513, 181, 151, 92, 286, 465], "num_ref_tokens": [111, 198, 114, 153, 127, 123, 98, 105, 115, 153, 108, 161, 154, 77, 106, 109, 97, 109, 124, 117, 99, 168, 121, 157, 105, 89, 124, 151, 92, 136, 79, 104, 78, 168, 124, 128, 87, 111, 42, 117, 94, 142, 145, 76, 179, 116, 131, 107, 107, 110, 138, 123, 137, 125, 140, 136, 134, 132, 112, 104, 155, 85, 98, 154, 90, 129, 152, 26, 111, 199, 121, 92, 121, 137, 109, 97, 96, 165, 87, 99, 108, 101, 136, 76, 179, 138, 129, 120, 147, 122, 114, 153, 108, 123, 153, 150, 131, 114, 141, 80]}, 5: {"num_pred_tokens": [326, 280, 213, 274, 217, 370, 334, 151, 377, 318, 181, 219, 187, 160, 334, 326, 364, 207, 103, 217, 363, 315, 318, 147, 217, 308, 333, 462, 374, 361, 366, 153, 513, 350, 296, 458, 147, 90, 337, 362, 110, 353, 133, 316, 348, 141, 149, 318, 158, 155, 165, 119, 400, 103, 326, 296, 149, 294, 164, 134, 148, 329, 392, 106, 22, 177, 187, 120, 382, 162, 96, 136, 116, 332, 297, 362, 352, 175, 121, 363, 374, 342, 335, 355, 382, 105, 352, 321, 243, 146, 133, 334, 225, 215, 146, 138, 98, 105, 496, 160], "num_ref_tokens": [145, 236, 174, 148, 150, 130, 128, 186, 129, 224, 113, 145, 137, 123, 134, 168, 151, 186, 161, 152, 114, 200, 119, 159, 109, 110, 195, 141, 156, 105, 97, 127, 130, 201, 114, 194, 153, 140, 138, 143, 163, 162, 172, 166, 189, 137, 151, 141, 176, 105, 207, 109, 209, 103, 145, 216, 143, 140, 172, 124, 114, 200, 149, 90, 138, 159, 99, 265, 126, 155, 159, 182, 174, 129, 166, 176, 184, 148, 152, 114, 200, 132, 149, 203, 149, 158, 172, 152, 117, 164, 187, 164, 174, 194, 129, 112, 112, 105, 117, 198]}}
df = pd.DataFrame(df1_data).T.reset_index().rename(columns={"index": "sentence_count"})
df_long_pred = df[["sentence_count", "num_pred_tokens"]].explode("num_pred_tokens").assign(token_type="Predicted").rename(columns={"num_pred_tokens": "tokens"})
df_long_ref = df[["sentence_count", "num_ref_tokens"]].explode("num_ref_tokens").assign(token_type="Reference").rename(columns={"num_ref_tokens": "tokens"})
df_long = pd.concat([df_long_pred, df_long_ref], ignore_index=True)
df_long["tokens"] = pd.to_numeric(df_long["tokens"])
mask = (df_long['sentence_count'] <= 5) & (df_long['sentence_count'] >= 4) & (df_long['tokens'] >= 180) & (df_long['token_type'] == "Predicted")
df_long = df_long.loc[~mask]

# Data 2: Metrics
data2 = {
    1: {'BLEU': 0.060145, 'ROUGE-L': 0.358415, 'Cosine': 0.896752, 'Pred Words': 25.4, 'Ref Words': 26.9},
    2: {'BLEU': 0.018446, 'ROUGE-L': 0.254710, 'Cosine': 0.828897, 'Pred Words': 49.5, 'Ref Words': 49.0},
    3: {'BLEU': 0.009736, 'ROUGE-L': 0.219589, 'Cosine': 0.830265, 'Pred Words': 57.5, 'Ref Words': 70.7},
    4: {'Pred Words': 66.0, 'Ref Words': 90.7},
    5: {'Pred Words': 80.0, 'Ref Words': 105.0}
}
df_metrics = pd.DataFrame(data2).T

# Nature/Science Styling
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.2,
})

colors = ["#E64B35", "#4DBBD5"] # Sci-kit color palette (NPG style)

# --- Plot 3: 2D KDE Hexbin/Density Contour Overlay ---
fig, ax = plt.subplots(figsize=(5.5, 5))

# We'll plot KDEs of sentence_count == 3 (where divergence starts getting strong)
df_sc3 = df_long[df_long['sentence_count'] == 3]

sns.kdeplot(
    data=df_sc3[df_sc3['token_type'] == 'Predicted'], x="tokens", 
    fill=True, alpha=0.5, linewidth=2, label="Predicted", color=colors[0], ax=ax, bw_adjust=0.3
)
sns.kdeplot(
    data=df_sc3[df_sc3['token_type'] == 'Reference'], x="tokens", 
    fill=True, alpha=0.5, linewidth=2, label="Reference", color=colors[1], ax=ax, bw_adjust=0.3
)

# Add rugs for exact data points
sns.rugplot(data=df_sc3[df_sc3['token_type'] == 'Predicted'], x="tokens", color=colors[0], ax=ax, height=0.05)
sns.rugplot(data=df_sc3[df_sc3['token_type'] == 'Reference'], x="tokens", color=colors[1], ax=ax, height=0.05)

ax.set_xlabel("Token Count")
ax.set_ylabel("Probability Density")
ax.set_title("Token Count Distribution", )
ax.set_ylim(0, 0.02)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 2. 设置左边框（Y轴）和底边框（X轴）的线宽
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

x_min, x_max = 0, 400
ax.spines['bottom'].set_bounds(0, 400)
ax.spines['left'].set_bounds(0, 0.02)

ax.yaxis.set_major_locator(MultipleLocator(0.01))


ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5)

# 3. 将边框向外推一点，离开数据区（这是形成独立括号感的关键）
# ax.spines['left'].set_position(('outward', 10))
# ax.spines['bottom'].set_position(('outward', 10))


ax.legend(frameon=False, loc='upper right')
plt.tight_layout()
fig.savefig("outputs/visual/num_tokens_recovered_from_compress_token.pdf", dpi=300)
plt.close(fig)

print("Plots generated successfully.")