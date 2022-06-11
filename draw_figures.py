import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from datetime import datetime

def convert2one_hot(labels, nclass, device):
    labels = labels.tolist()
    one_hot = []
    for l in range(len(labels)):
        cur_sample = []
        for i in range(nclass):
            if i == labels[l]:
                cur_sample.append(1)
            else:
                cur_sample.append(0)
        one_hot.append(cur_sample)
    return torch.Tensor(one_hot).to(device)


def calc_bins(labels_oneh, preds):
  # Assign each prediction to a bin
  num_bins = 10 #change bin number ECE will be different
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]
  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(labels_oneh, preds):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(labels_oneh, preds)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)
  return ECE, MCE


def draw_reliability_graph(labels_oneh, preds, dataset_string, model_string, task_type):
  ECE, MCE = get_metrics(labels_oneh, preds)
  bins, _, bin_accs, _, _ = calc_bins(labels_oneh, preds)

  fig = plt.figure(figsize=(6, 4))
  plt.rcParams['axes.labelweight'] = 'bold'
  plt.rcParams["font.weight"] = "bold"
  
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence',fontsize=14)
  plt.ylabel('Accuracy',fontsize=14)
  # Create grid
  ax.set_axisbelow(True) 
  # ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins, width=0.09, alpha=0.7, color='lightcoral', label='Expected') 

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.09, alpha=0.7, color='dodgerblue', label='Outputs') 
  plt.plot([0,1],[0,1], '--', c='k', linewidth=1)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box') #add to get square

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  # plt.legend(handles=[ECE_patch, MCE_patch])
  plt.tick_params(labelsize=13)
  plt.legend(fontsize=14)
  # plt.title(model_string + ' ' +dataset_string, fontsize=16, fontweight="bold")
  # plt.show()
  today_date = datetime.today().strftime('%Y-%m-%d')

  plt.savefig(today_date + '_ECE_plot_'+ model_string + '_' + dataset_string + '_' + task_type +'.png', bbox_inches='tight',format='png', dpi=300,
                pad_inches=0)
  
  return ECE


def double_figure(loss_log, acc_log):
    total_len = len(loss_log)
    x_plot = np.array([i for i in range(total_len)],dtype=np.float64)

    fig = plt.figure(figsize=(6, 6), 
                    edgecolor='black')
    lg_font_mine = {'family': 'Arial', 'weight': 'bold', 'size': 15,}

    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.axis('off')
    plt.xlim(0, total_len)
    plt.xticks(np.arange(0, total_len, 50))
    plt.xlabel("Training Epochs",
              fontsize=15,
              family='Arial',
              weight='bold')

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.set_xlabel("Training Epochs",
              fontsize=15,
              # family='Arial',
              weight='bold')
    
    curve_mean = ax1.plot(x_plot, loss_log, label = 'Loss values', color='black',lw=2,alpha = 1.) #1685A9
    ax1.set_ylabel("Loss values",
                  fontsize=17,
              weight='bold')

    curve_psnr = ax2.plot(np.array(acc_log), label = 'Accuracy', color='#F36838', lw=2)
    ax2.set_ylabel("Accuracy",
                  fontsize=17,
              weight='bold')
    
    curve_all = curve_mean + curve_psnr
    label_all = [l.get_label() for l in curve_all]
    ax2.legend(curve_all, label_all, prop= {'family': 'Arial', 'weight': 'bold', 'size': 17,},
              edgecolor='gray',
              loc=(0.45, 0.4), # default upper right: (1,1)
              ncol=1)

    for label in ax1.get_xticklabels():
        label.set_fontsize(14)
    for label in ax1.get_yticklabels():
        label.set_fontsize(14)
    for label in ax2.get_yticklabels():
        label.set_fontsize(14)

    ax1.grid(alpha=0.6)
    plt.savefig('Double_axis.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.show()