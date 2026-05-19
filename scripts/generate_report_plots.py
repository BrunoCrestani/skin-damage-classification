#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Professional academic styling
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'grid.color': '#dddddd',
        'grid.linestyle': '--'
    })
    
    # ----------------------------------------------------
    # Plot 1: Model Comparison (Accuracy, F1-Macro, AUC-ROC)
    # ----------------------------------------------------
    models = [
        "Tabular: LR", "Tabular: RF", 
        "Image: LR", "Image: KNN", "Image: RF",
        "Multimodal: Late Fusion"
    ]
    
    accuracy = [0.5163, 0.6001, 0.7512, 0.6693, 0.6138, 0.7505]
    f1_macro = [0.2762, 0.3611, 0.5898, 0.5065, 0.4042, 0.5824]
    auc_roc = [0.8655, 0.8929, 0.9298, 0.7859, 0.9136, 0.9341]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    rects1 = ax.bar(x - width, accuracy, width, label='Acurácia', color='#1f77b4', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, f1_macro, width, label='F1-Macro', color='#ff7f0e', edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, auc_roc, width, label='AUC-ROC Macro', color='#2ca02c', edgecolor='black', linewidth=0.8)
    
    ax.set_title("Comparativo Geral de Desempenho dos Modelos (Conjunto de Teste)", pad=20, weight='bold')
    ax.set_xlabel("Configuração de Modelo", labelpad=12)
    ax.set_ylabel("Pontuação / Score (0.0 - 1.0)", labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Value labels function
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    chart1_path = output_dir / "model_comparison_chart.png"
    plt.savefig(chart1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart1_path}")
    
    # ----------------------------------------------------
    # Plot 2: Imbalance Study (F1-Macro vs Accuracy)
    # ----------------------------------------------------
    strategies = ["Sem Correção", "Sensível ao Custo (Pesos)", "Sobreamostragem (ROS)"]
    f1_val = [0.3767, 0.3812, 0.3646]
    acc_val = [0.7224, 0.6232, 0.6052]
    
    x_imb = np.arange(len(strategies))
    width_imb = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    rects1_imb = ax.bar(x_imb - width_imb/2, acc_val, width_imb, label='Acurácia', color='#4a90e2', edgecolor='black', linewidth=0.8)
    rects2_imb = ax.bar(x_imb + width_imb/2, f1_val, width_imb, label='F1-Macro', color='#e24a4a', edgecolor='black', linewidth=0.8)
    
    ax.set_title("Estudo do Desbalanceamento de Classes (Random Forest Tabular na Validação)", pad=20, weight='bold')
    ax.set_xlabel("Método de Tratamento de Imbalance", labelpad=12)
    ax.set_ylabel("Pontuação / Score (0.0 - 1.0)", labelpad=12)
    ax.set_xticks(x_imb)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(loc="upper right")
    
    def autolabel_imb(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel_imb(rects1_imb)
    autolabel_imb(rects2_imb)
    
    plt.tight_layout()
    chart2_path = output_dir / "imbalance_study_chart.png"
    plt.savefig(chart2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart2_path}")

if __name__ == "__main__":
    main()
