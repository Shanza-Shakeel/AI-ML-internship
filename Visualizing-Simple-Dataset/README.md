# Iris Dataset Exploration Project

## Task Objective
Perform comprehensive exploratory data analysis (EDA) of the Iris dataset to:
- Analyze feature distributions and relationships
- Identify distinguishing characteristics between Iris species
- Document insights for machine learning model development

## Dataset Used
**Iris Flower Dataset** (UCI Machine Learning Repository)
- **Samples**: 150 (50 per species)
- **Features**:
  - Sepal length (cm)
  - Sepal width (cm) 
  - Petal length (cm)
  - Petal width (cm)
- **Target Variable**: Species (Iris-setosa, Iris-versicolor, Iris-virginica)

## Models Applied
### Statistical Analysis
- Descriptive statistics (mean, median, standard deviation)
- Correlation analysis (Pearson coefficient)
- Outlier detection using IQR method

### Visualization Techniques
- Distribution plots: Histograms, Violin plots
- Comparative plots: Box plots, Pair plots
- Relationship plots: Scatter matrix
- Correlation heatmap

## Key Results and Findings

### Species Differentiation
| Characteristic   | Iris-setosa | Iris-versicolor | Iris-virginica |
|------------------|-------------|------------------|----------------|
| Avg Petal Length | 1.46 cm     | 4.26 cm          | 5.55 cm        |
| Avg Petal Width  | 0.24 cm     | 1.33 cm          | 2.03 cm        |
| Avg Sepal Length | 5.01 cm     | 5.94 cm          | 6.59 cm        |
| Avg Sepal Width  | 3.42 cm     | 2.77 cm          | 2.97 cm        |

### Significant Findings
1. **Petal measurements** show the clearest separation between species
2. **Setosa** is easily identifiable with significantly smaller petals
3. **Versicolor vs Virginica** can be distinguished by petal dimensions despite some overlap
4. **Strongest correlation** observed between petal length and width (r = 0.96)

## How to Run
### Requirements
```bash
pip install pandas numpy matplotlib seaborn jupyter

## 🚀 How to Run Locally
Clone this repository:
   ```bash
 git clone https://github.com/Shanza-Shakeel/AI-ML-internship/Visualizing-Simple-Dataset.git
   cd Visualizing-Simple-Dataset