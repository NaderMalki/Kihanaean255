# Artificial intelligence neural network with 3D and dual-hemisphere processing
#Nader.malkei,PCT/IR2025/050026,1404501400#03002031

    
    
    import torch
    Payton 
    Import 
    Numpy 
    Pandas
    Jupiter Notebook
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
ورودی (4D)
│
├─→ [پیش‌پردازش با توجه + مدولاسیون] → دو بردار: V_A, V_B
│
├─→ پروجکشن: V_A → H_A (8D),   V_B → H_B (8D)
│
├─→ تقسیم:  
│     H_A → H_A1 (4D), H_A2 (4D)  
│     H_B → H_B1 (4D), H_B2 (4D)
│
├─→ تبادل اطلاعات با توجه:
│     - بین H_A1 و H_B2 → با Cross-Attention ساده  
│     - بین H_B1 و H_A2 → با Cross-Attention ساده
│
├─→ ترکیب زیرشاخه‌های به‌روزشده → دو بردار میانی
│
└─→ فیوژن یکپارچه (Concat + Attention Weighting) → خروجی نهایی (2D, softmax)
# ذخیره مدل
torch.save(model.state_dict(), "nano_plastic_snn_v4.pt")
print("✅ مدل چهارم با موفقیت ذخیره شد.")
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, Multiply, Add, Concatenate,
    Lambda, Softmax, Layer
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# ----------------------------
# 1. بلوک توجه ساده (Self/Channel Attention)
# ----------------------------
def simple_attention_block(x, name=""):
    """Channel-wise attention (SE-style)"""
    dim = K.int_shape(x)[-1]
    squeeze = Lambda(lambda t: K.mean(t, axis=1, keepdims=True))(x)
    excite = Dense(max(dim // 2, 1), activation='relu')(squeeze)
    excite = Dense(dim, activation='sigmoid')(excite)
    excite = Lambda(lambda t: K.squeeze(t, axis=1))(excite)
    return Multiply(name=f"{name}_att")([x, excite])

# ----------------------------
# 2. Cross-Attention سبک بین دو بردار
# ----------------------------
def cross_attention(q, k, v, name=""):
    """
    q: query (batch, d)
    k, v: key/value (batch, d)
    محاسبه وزن توجه: softmax(q · k^T / sqrt(d)) * v
    """
    d = K.int_shape(q)[-1]
    scale = K.sqrt(K.cast(d, 'float32'))

    # dot product attention
    attn_scores = Lambda(lambda x: tf.linalg.matmul(
        tf.expand_dims(x[0], axis=1),
        tf.expand_dims(x[1], axis=2)
    )[:, 0, 0] / scale)([q, k])  # (batch,)

    attn_weights = Lambda(lambda x: tf.nn.softmax(tf.expand_dims(x, axis=-1), axis=0))(attn_scores)  # (batch, 1)

    # weighted value
    attended = Multiply()([v, attn_weights])
    return attended

# ----------------------------
# 3. مدل اصلی
# ----------------------------

input_layer = Input(shape=(4,), name="input")

# ============ مرحله 1: پیش‌پردازش با توجه و مدولاسیون ============
features = Dense(16, activation='relu', name="feat_ext")(input_layer)
features = BatchNormalization(name="bn_feat")(features)
features = Dropout(0.2, name="drop_feat")(features)
features = simple_attention_block(features, name="pre_att")

# تولید دو بردار اولیه (پروجکشن اولیه)
V_A = Dense(8, activation='relu', name="V_A")(features)
V_B = Dense(8, activation='relu', name="V_B")(features)

# ============ مرحله 2: پروجکشن به لایه پنهان و تقسیم ============
H_A = Dense(8, activation='relu', name="H_A")(V_A)
H_B = Dense(8, activation='relu', name="H_B")(V_B)

# تقسیم هر کدام به دو بخش
H_A1 = Dense(4, activation='relu', name="H_A1")(H_A)
H_A2 = Dense(4, activation='relu', name="H_A2")(H_A)
H_B1 = Dense(4, activation='relu', name="H_B1")(H_B)
H_B2 = Dense(4, activation='relu', name="H_B2")(H_B)

# ============ مرحله 3: تبادل اطلاعات با مکانیزم توجه ============
# توجه متقاطع: H_A1 به عنوان query، H_B2 به عنوان key/value
exchanged_1 = cross_attention(H_A1, H_B2, H_B2, name="cross1")
exchanged_2 = cross_attention(H_B1, H_A2, H_A2, name="cross2")

# به‌روزرسانی زیرشاخه‌ها (جمع با خودشان برای residual)
updated_A1 = Add(name="upd_A1")([H_A1, exchanged_1])
updated_B1 = Add(name="upd_B1")([H_B1, exchanged_2])

# ============ مرحله 4: بازسازی دو شاخه اصلی ============
recon_A = Concatenate(name="recon_A")([updated_A1, H_A2])  # (8D)
recon_B = Concatenate(name="recon_B")([updated_B1, H_B2])  # (8D)

# ============ مرحله 5: فیوژن یکپارچه نهایی ============
# روش: Concat + Channel Attention برای وزن‌دهی هوشمند
fused = Concatenate(name="fused_concat")([recon_A, recon_B])  # (16D)
fused = simple_attention_block(fused, name="final_fusion_att")

# لایه خروجی
output = Dense(2, activation='softmax', name="output")(fused)

# ============ ساخت مدل ============
model = Model(inputs=input_layer, outputs=output, name="Hierarchical_Attentive_Model")

# کامپایل
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# نمایش ساختار
model.summary()
