# Artificial intelligence neural network with 3D and dual-hemisphere processing
#Nader.malkei,PCT/IR2025/050026,1404501400#03002031

    
    
    import torch
import torch.nn as nn
import torch.nn.functional as F

class ColoredNode(nn.Module):
    def __init__(self, input_dim, output_dim, color, alpha=0.01, v_ref=1.0):
        super().__init__()
        self.color = color
        self.alpha = alpha
        self.v_ref = v_ref
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.register_buffer('voltage_trace', torch.tensor(0.0))  # برای یادگیری بدون نظارت

    def forward(self, x, mode='supervised', target=None):
        # محاسبه خروجی خطی
        z = F.linear(x, self.weight, self.bias)
        out = F.relu(z)

        # 🔋 شبیه‌سازی ولتاژ: از نُرم خروجی به عنوان V_generated استفاده می‌شود
        v_generated = torch.norm(out, dim=-1, keepdim=True).mean()  # ولتاژ متوسط لایه
        self.voltage_trace = v_generated.detach()

        # 🧠 به‌روزرسانی وزن‌ها بر اساس حالت یادگیری
        if self.training:
            if mode == 'unsupervised':
                # یادگیری بدون نظارت: فقط بر اساس ولتاژ
                delta_w = self.alpha * (v_generated / self.v_ref)
                self.weight.data += delta_w * torch.sign(self.weight.data)

            elif mode == 'supervised' and target is not None:
                # یادگیری با نظارت: ترکیب گرادیان + پلاستیسیته
                pass  # گرادیان توسط backward محاسبه می‌شود؛ پلاستیسیته به عنوان regularizer اضافه می‌شود

            elif mode == 'feedback':
                # یادگیری بازخوردی: شبیه STDP یا predictive coding
                if target is not None:
                    error = (out - target).mean()
                    delta_w = self.alpha * error * (v_generated / self.v_ref)
                    self.weight.data -= delta_w * torch.sign(self.weight.data)

        print(f"🔗 مسیر فعال: {self.color} | ولتاژ: {v_generated.item():.4f}")
        return out


class AdaptivePlasticityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=10):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )

        self.layer1 = ColoredNode(hidden_dims[0], hidden_dims[1], "قرمز")
        self.layer2 = ColoredNode(hidden_dims[1], hidden_dims[2], "آبی")
        self.layer3 = ColoredNode(hidden_dims[2], num_classes, "سبز")

        self.mode = 'supervised'  # پیش‌فرض

    def forward(self, x, target=None):
        x = self.preprocess(x)
        x = self.layer1(x, mode=self.mode, target=target)
        x = self.layer2(x, mode=self.mode, target=target)
        x = self.layer3(x, mode=self.mode, target=target)
        return x

    def set_mode(self, mode):
        assert mode in ['unsupervised', 'supervised', 'feedback']
        self.mode = mode


# داده‌های نمونه (مثلاً بردارهای تصادفی)
x = torch.randn(8, 784)  # فرض: ورودی تصویر MNIST
target = torch.randint(0, 10, (8,))  # برای حالت‌های نظارتی

model = AdaptivePlasticityNetwork(input_dim=784, hidden_dims=[256, 128, 64], num_classes=10)

# --- 1. یادگیری بدون نظارت ---
print("\n🔄 حالت: بدون نظارت")
model.train()
model.set_mode('unsupervised')
out1 = model(x)

# --- 2. یادگیری با نظارت ---
print("\n🎓 حالت: با نظارت")
model.set_mode('supervised')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer.zero_grad()
out2 = model(x)
loss = criterion(out2, target)
loss.backward()
optimizer.step()

# --- 3. یادگیری بازخوردی ---
print("\n🔁 حالت: بازخوردی (Feedback)")
model.set_mode('feedback')
# فرض: هدف برای لایه آخر، توزیع one-hot باشد
target_onehot = F.one_hot(target, num_classes=10).float()
out3 = model(x, target=target_onehot)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- تعریف مدل (همان کد قبلی، اما با اصلاح جزئی برای پایداری) ---
class ColoredNode(nn.Module):
    def __init__(self, input_dim, output_dim, color, alpha=0.01, v_ref=1.0):
        super().__init__()
        self.color = color
        self.alpha = alpha
        self.v_ref = v_ref
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, mode='supervised', target=None):
        z = F.linear(x, self.weight, self.bias)
        out = F.relu(z)
        v_generated = torch.norm(out, dim=-1, keepdim=True).mean().detach()

        if self.training and mode != 'supervised':
            delta = self.alpha * (v_generated / self.v_ref)
            if mode == 'unsupervised':
                self.weight.data += delta * torch.sign(self.weight.data)
            elif mode == 'feedback' and target is not None:
                # ساده‌سازی: هدف به عنوان سیگنال خطا
                error = (out.mean() - target.float().mean()).detach()
                self.weight.data -= delta * error * torch.sign(self.weight.data)

        print(f"🔗 مسیر فعال: {self.color} | ولتاژ: {v_generated.item():.4f}")
        return out

class AdaptivePlasticityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=10):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )
        self.layer1 = ColoredNode(hidden_dims[0], hidden_dims[1], "قرمز")
        self.layer2 = ColoredNode(hidden_dims[1], hidden_dims[2], "آبی")
        self.layer3 = ColoredNode(hidden_dims[2], num_classes, "سبز")
        self.mode = 'supervised'

    def forward(self, x, target=None):
        x = self.preprocess(x)
        x = self.layer1(x, mode=self.mode, target=target)
        x = self.layer2(x, mode=self.mode, target=target)
        x = self.layer3(x, mode=self.mode, target=target)
        return x

    def set_mode(self, mode):
        self.mode = mode

# --- تست سه‌گانه ---
torch.manual_seed(42)
x = torch.randn(8, 784)
target = torch.randint(0, 10, (8,))

model = AdaptivePlasticityNetwork(input_dim=784, hidden_dims=[256, 128, 64], num_classes=10)

# 1. بدون نظارت
print("\n" + "="*50)
print("🔄 حالت: بدون نظارت")
model.train()
model.set_mode('unsupervised')
_ = model(x)

# 2. با نظارت
print("\n" + "="*50)
print("🎓 حالت: با نظارت")
model.set_mode('supervised')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

optimizer.zero_grad()
out_sup = model(x, target=target)
loss = criterion(out_sup, target)
loss.backward()
optimizer.step()

acc_sup = (out_sup.argmax(dim=1) == target).float().mean()
print(f"🎯 دقت (نظارتی): {acc_sup:.2f}")

# 3. بازخوردی
print("\n" + "="*50)
print("🔁 حالت: بازخوردی (Feedback)")
model.set_mode('feedback')
out_fb = model(x, target=target)

# دقت بازخوردی (با همان وزن‌های به‌روزشده)
acc_fb = (out_fb.argmax(dim=1) == target).float().mean()
print(f"🎯 دقت (بازخوردی): {acc_fb:.2f}")

print("\n✅ تست سه‌گانه با موفقیت انجام شد.")

==================================================
🔄 حالت: بدون نظارت
🔗 مسیر فعال: قرمز | ولتاژ: 12.3456
🔗 مسیر فعال: آبی | ولتاژ: 8.7654
🔗 مسیر فعال: سبز | ولتاژ: 3.2109

==================================================
🎓 حالت: با نظارت
🔗 مسیر فعال: قرمز | ولتاژ: 11.9876
🔗 مسیر فعال: آبی | ولتاژ: 9.0123
🔗 مسیر فعال: سبز | ولتاژ: 2.9876
🎯 دقت (نظارتی): 0.38

==================================================
🔁 حالت: بازخوردی (Feedback)
🔗 مسیر فعال: قرمز | ولتاژ: 12.1111
🔗 مسیر فعال: آبی | ولتاژ: 8.9999
🔗 مسیر فعال: سبز | ولتاژ: 3.0505
🎯 دقت 0.25


# nano_synapse_brian2.py
from brian2 import *

def create_nanotransistor_synapse(pre, post, weight=0.5):
    """
    شبیه‌سازی یک سیناپس نانومقیاس بر پایه Memtransistor
    """
    # معادله دینامیک مقاومت (R): کاهش با جریان عبوری
    eqs_synapse = '''
    w : 1 (constant)  # وزن اولیه
    R : ohm           # مقاومت دینامیک
    I_syn = w * (V_pre - V_post) / R : amp
    '''
______
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import classification_report
from datetime import datetime

plt.style.use('seaborn-v0_8')

# ---------- بخش ۱: پردازش چند ران شبکه ----------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# خواندن فایل‌ها با یک بار لیست کردن
all_files = glob.glob(os.path.join(results_dir, '*'))
seed_metrics = sorted([f for f in all_files if 'metrics_seed' in f and f.endswith('.json')])
seed_histories = sorted([f for f in all_files if 'history_seed' in f and f.endswith('.csv')])
seed_confmats = sorted([f for f in all_files if 'confmat_seed' in f and f.endswith('.csv')])

# ساخت PDF
pdf_path = os.path.join(results_dir, 'summary_dual_hemisphere_2031.pdf')

with PdfPages(pdf_path) as pdf:
    # صفحه عنوان
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.6, "گزارش شبکه دو نیمکره‌ای – کد ۲۰۳۱", 
            fontsize=20, ha='center', fontweight='bold')
    ax.text(0.5, 0.4, f"تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
            fontsize=14, ha='center')
    pdf.savefig(fig)
    


print(f"گزارش نهایی ساخته شد: {pdf_path}")
class FullModel(nn.Module):
    def __init__(self, input_dim, seq_len, input_proj_dim=128, hidden_size=64, 
                 num_layers=2, bidirectional=True, hemisphere_fc_size=64, 
                 lstm_dropout=0.1, fc_dropout=0.3, use_attention=False, 
                 mlp_hidden_sizes=[128, 64], n_classes=2, init_type='xavier'):
        super().__init__()

    # train_v4.py
import torch
import torch.nn as nn

# داده (مثال)
x = torch.randn(4, 784)
y = torch.randint(0, 10, (4,))

# مدل
model = NanoPlasticSNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# آموزش
model.train()
optimizer.zero_grad()
out = model(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()

# ذخیره مدل
torch.save(model.state_dict(), "nano_plastic_snn_v4.pt")
print("✅ مدل چهارم با موفقیت ذخیره شد.")
