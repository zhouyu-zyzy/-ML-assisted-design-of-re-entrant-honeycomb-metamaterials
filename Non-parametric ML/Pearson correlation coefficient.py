import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import os

file_path = r"E:\001-stimulation and procedure\03-PYTHON Files\9-SISSO\07-pearson/PS.csv"

try:
    data = pd.read_csv(file_path, encoding='gbk')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='utf-8')

correlation_matrix = data.corr(method='pearson')

mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=0)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.size'] = 9
plt.figure(figsize=(6 / 2.54, 5 / 2.54), facecolor='none')
labels = [re.sub(r'θ(\d+)', r'$\theta_\1$', col) for col in correlation_matrix.columns]

sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            cbar=True, 
            mask=~mask,      
            square=True, 
            cbar_kws={'aspect': 40, 'pad': 0.02},  
            xticklabels=labels, 
            yticklabels=labels)


plt.title("Pearson Correlation Coefficient", fontsize=9)
plt.xticks(rotation=45)
plt.yticks(rotation=45)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"PS.tif"


save_dir = r"E:\001-stimulation and procedure\03-PYTHON Files\9-SISSO\07-pearson"
os.makedirs(save_dir, exist_ok=True) 
save_path = os.path.join(save_dir, filename)
plt.tight_layout()
plt.savefig(save_path, format='tiff', dpi=1200, transparent=True)
plt.show()
