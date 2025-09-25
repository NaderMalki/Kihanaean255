# Artificial intelligence neural network with 3D and dual-hemisphere processing
Nader.malkei ,PCT/IR2025/050026,140450140003002031iran
    def __init__(self, input_dim, seq_len, input_proj_dim=128, hidden_size=64, 
                 num_layers=2, bidirectional=True, hemisphere_fc_size=64, 
                 lstm_dropout=0.1, fc_dropout=0.3, use_attention=False, 
                 mlp_hidden_sizes=[128, 64], n_classes=2, init_type='xavier'):
        super().__init__()

        # پیش‌پردازش / projection ورودی
        self.projector = InputProjector(input_dim, input_proj_dim, use_layernorm=True) \
            if input_dim != input_proj_dim else nn.Identity()

        # بلوک هماهنگ دو نیمکره
        self.dual = CoordinatedDualHemisphere(
            input_size=input_proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            fc_size=hemisphere_fc_size,
            lstm_dropout=lstm_dropout,
            fc_dropout=fc_dropout,
            use_attention=use_attention
        )

        # MLP برای طبقه‌بندی نهایی
        combined_size = hemisphere_fc_size * 2  # برای دو نیمکره
        mlp_layers = []
        in_features = combined_size

        for hidden_size in mlp_hidden_sizes:
            mlp_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            ])
            in_features = hidden_size

        mlp_layers.append(nn.Linear(in_features, n_classes))
        self.mlp = nn.Sequential(*mlp_layers)

        # مقداردهی اولیه وزن‌ها
        self._init_weights(init_type)

    def _init_weights(self, init_type):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_left, x_right):
        # Project input
        x_left_proj = self.projector(x_left)
        x_right_proj = self.projector(x_right)

        # Process through dual hemispheres
        features = self.dual(x_left_proj, x_right_proj)

        # Final classification
        output = self.mlp(features)

        return output
        # test_piezoelectric_report.py
import unittest
import os
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Import the functions from the main code (after refactoring)
from main_code import (
    create_title_page,
    process_seed_data,
    simulate_piezoelectric_sensor,
    create_innovation_page,
    generate_report
)

class TestPiezoelectricReport(unittest.TestCase):

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.test_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Create test data files
        self.create_test_files()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        plt.close('all')

    def create_test_files(self):
        """Create test JSON, CSV files for testing"""
        # Create metrics JSON files
        for i in range(2):
            metrics_data = {
                'test_accuracy': 0.85 + i*0.05,
                'f1_score': 0.82 + i*0.04
            }
            with open(os.path.join(self.results_dir, f'metrics_seed{i}.json'), 'w') as f:
                json.dump(metrics_data, f)

        # Create history CSV files
        for i in range(2):
            history_data = {
                'epoch': range(10),
                'accuracy': [0.1*j for j in range(10)],
                'val_accuracy': [0.08*j for j in range(10)]
            }
            df = pd.DataFrame(history_data)
            df.to_csv(os.path.join(self.results_dir, f'history_seed{i}.csv'), index=False)

        # Create confusion matrix CSV files
        for i in range(2):
            confmat_data = np.array([[5, 2], [1, 7]])
            df = pd.DataFrame(confmat_data)
            df.to_csv(os.path.join(self.results_dir, f'confmat_seed{i}.csv'), index=False)

    def test_create_title_page(self):
        """Test title page creation"""
        fig = create_title_page()
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)

    def test_process_seed_data(self):
        """Test processing of seed data"""
        metrics_file = os.path.join(self.results_dir, 'metrics_seed0.json')
        hist_file = os.path.join(self.results_dir, 'history_seed0.csv')
        conf_file = os.path.join(self.results_dir, 'confmat_seed0.csv')

        # Mock PdfPages to capture saved figures
        with patch('matplotlib.backends.backend_pdf.PdfPages') as mock_pdf:
            mock_pdf_instance = MagicMock()
            mock_pdf.return_value.__enter__.return_value = mock_pdf_instance

            process_seed_data(metrics_file, hist_file, conf_file, mock_pdf_instance)

            # Check that savefig was called 3 times (curves, confmat, text)
            self.assertEqual(mock_pdf_instance.savefig.call_count, 3)

    def test_simulate_piezoelectric_sensor(self):
        """Test piezoelectric sensor simulation"""
        # Mock PdfPages
        with patch('matplotlib.backends.backend_pdf.PdfPages') as mock_pdf:
            mock_pdf_instance = MagicMock()
            mock_pdf.return_value.__enter__.return_value = mock_pdf_instance

            fig = simulate_piezoelectric_sensor()

            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 3)  # Should have 3 subplots

    def test_simulate_piezoelectric_calculations(self):
        """Test the calculations in piezoelectric simulation"""
        # Test parameters
        params = {
            'd33': 2.5e-10,
            'spring_k': 1000,
            'damping_c': 10,
            'capacitance_C': 1e-6,
            'radius': 0.01
        }

        # Test data
        time = np.linspace(0, 1, 100)
        x = 0.01 * np.sin(2 * np.pi * 5 * time)
        dx_dt = np.gradient(x, time)

        # Calculations
        cross_section = np.pi * params['radius'] ** 2
        F_mech = params['spring_k'] * x + params['damping_c'] * dx_dt
        sigma = F_mech / cross_section
        Q = params['d33'] * F_mech
        V = Q / params['capacitance_C']

        # Assertions
        self.assertEqual(len(V), 100)
        self.assertEqual(len(F_mech), 100)
        self.assertEqual(len(sigma), 100)
        self.assertTrue(np.all(np.isfinite(V)))
        self.assertTrue(np.all(np.isfinite(F_mech)))
        self.assertTrue(np.all(np.isfinite(sigma)))

    def test_create_innovation_page(self):
        """Test innovation page creation"""
        fig = create_innovation_page()
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)

    @patch('matplotlib.backends.backend_pdf.PdfPages')
    @patch('glob.glob')
    def test_generate_report_integration(self, mock_glob, mock_pdf):
        """Test the complete report generation process"""
        # Mock glob to return our test files
        mock_glob.side_effect = [
            sorted([os.path.join(self.results_dir, f'metrics_seed{i}.json') for i in range(2)]),
            sorted([os.path.join(self.results_dir, f'history_seed{i}.csv') for i in range(2)]),
            sorted([os.path.join(self.results_dir, f'confmat_seed{i}.csv') for i in range(2)])
        ]

        # Mock PdfPages
        mock_pdf_instance = MagicMock()
        mock_pdf.return_value.__enter__.return_value = mock_pdf_instance

        # Generate report
        pdf_path = generate_report(self.results_dir)

        # Verify PDF was created
        self.assertTrue(pdf_path.endswith('summary_dual_hemisphere_2031.pdf'))

        # Verify savefig was called multiple times (title + 2 seeds * 3 + sensor + innovations)
        expected_calls = 1 + (2 * 3) + 1 + 1  # title + seeds + sensor + innovations
        self.assertEqual(mock_pdf_instance.savefig.call_count, expected_calls)

    def test_file_reading(self):
        """Test that files can be read correctly"""
        # Test JSON reading
        with open(os.path.join(self.results_dir, 'metrics_seed0.json'), 'r') as f:
            metrics = json.load(f)
        self.assertIn('test_accuracy', metrics)
        self.assertIn('f1_score', metrics)

        # Test CSV reading
        history = pd.read_csv(os.path.join(self.results_dir, 'history_seed0.csv'))
        self.assertIn('epoch', history.columns)
        self.assertIn('accuracy', history.columns)
        self.assertIn('val_accuracy', history.columns)

        confmat = pd.read_csv(os.path.join(self.results_dir, 'confmat_seed0.csv'), index_col=0)
        self.assertEqual(confmat.shape, (2, 2))

if __name__ == '__main__':
    unittest.main()
    # test_piezoelectric_report.py
import unittest
import os
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Import the functions from the main code (after refactoring)
from main_code import (
    create_title_page,
    process_seed_data,
    simulate_piezoelectric_sensor,
    create_innovation_page,
    generate_report
)

class TestPiezoelectricReport(unittest.TestCase):

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.test_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Create test data files
        self.create_test_files()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        plt.close('all')

    def create_test_files(self):
        """Create test JSON, CSV files for testing"""
        # Create metrics JSON files
        for i in range(2):
            metrics_data = {
                'test_accuracy': 0.85 + i*0.05,
                'f1_score': 0.82 + i*0.04
            }
            with open(os.path.join(self.results_dir, f'metrics_seed{i}.json'), 'w') as f:
                json.dump(metrics_data, f)

        # Create history CSV files
        for i in range(2):
            history_data = {
                'epoch': range(10),
                'accuracy': [0.1*j for j in range(10)],
                'val_accuracy': [0.08*j for j in range(10)]
            }
            df = pd.DataFrame(history_data)
            df.to_csv(os.path.join(self.results_dir, f'history_seed{i}.csv'), index=False)

        # Create confusion matrix CSV files
        for i in range(2):
            confmat_data = np.array([[5, 2], [1, 7]])
            df = pd.DataFrame(confmat_data)
            df.to_csv(os.path.join(self.results_dir, f'confmat_seed{i}.csv'), index=False)

    def test_create_title_page(self):
        """Test title page creation"""
        fig = create_title_page()
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)

    def test_process_seed_data(self):
        """Test processing of seed data"""
        metrics_file = os.path.join(self.results_dir, 'metrics_seed0.json')
        hist_file = os.path.join(self.results_dir, 'history_seed0.csv')
        conf_file = os.path.join(self.results_dir, 'confmat_seed0.csv')

        # Mock PdfPages to capture saved figures
        with patch('matplotlib.backends.backend_pdf.PdfPages') as mock_pdf:
            mock_pdf_instance = MagicMock()
            mock_pdf.return_value.__enter__.return_value = mock_pdf_instance

            process_seed_data(metrics_file, hist_file, conf_file, mock_pdf_instance)

            # Check that savefig was called 3 times (curves, confmat, text)
            self.assertEqual(mock_pdf_instance.savefig.call_count, 3)

    def test_simulate_piezoelectric_sensor(self):
        """Test piezoelectric sensor simulation"""
        # Mock PdfPages
        with patch('matplotlib.backends.backend_pdf.PdfPages') as mock_pdf:
            mock_pdf_instance = MagicMock()
            mock_pdf.return_value.__enter__.return_value = mock_pdf_instance

            fig = simulate_piezoelectric_sensor()

            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 3)  # Should have 3 subplots

    def test_simulate_piezoelectric_calculations(self):
        """Test the calculations in piezoelectric simulation"""
        # Test parameters
        params = {
            'd33': 2.5e-10,
            'spring_k': 1000,
            'damping_c': 10,
            'capacitance_C': 1e-6,
            'radius': 0.01
        }

        # Test data
        time = np.linspace(0, 1, 100)
        x = 0.01 * np.sin(2 * np.pi * 5 * time)
        dx_dt = np.gradient(x, time)

        # Calculations
        cross_section = np.pi * params['radius'] ** 2
        F_mech = params['spring_k'] * x + params['damping_c'] * dx_dt
        sigma = F_mech / cross_section
        Q = params['d33'] * F_mech
        V = Q / params['capacitance_C']

        # Assertions
        self.assertEqual(len(V), 100)
        self.assertEqual(len(F_mech), 100)
        self.assertEqual(len(sigma), 100)
        self.assertTrue(np.all(np.isfinite(V)))
        self.assertTrue(np.all(np.isfinite(F_mech)))
        self.assertTrue(np.all(np.isfinite(sigma)))

    def test_create_innovation_page(self):
        """Test innovation page creation"""
        fig = create_innovation_page()
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)

    @patch('matplotlib.backends.backend_pdf.PdfPages')
    @patch('glob.glob')
    def test_generate_report_integration(self, mock_glob, mock_pdf):
        """Test the complete report generation process"""
        # Mock glob to return our test files
        mock_glob.side_effect = [
            sorted([os.path.join(self.results_dir, f'metrics_seed{i}.json') for i in range(2)]),
            sorted([os.path.join(self.results_dir, f'history_seed{i}.csv') for i in range(2)]),
            sorted([os.path.join(self.results_dir, f'confmat_seed{i}.csv') for i in range(2)])
        ]

        # Mock PdfPages
        mock_pdf_instance = MagicMock()
        mock_pdf.return_value.__enter__.return_value = mock_pdf_instance

        # Generate report
        pdf_path = generate_report(self.results_dir)

        # Verify PDF was created
        self.assertTrue(pdf_path.endswith('summary_dual_hemisphere_2031.pdf'))

        # Verify savefig was called multiple times (title + 2 seeds * 3 + sensor + innovations)
        expected_calls = 1 + (2 * 3) + 1 + 1  # title + seeds + sensor + innovations
        self.assertEqual(mock_pdf_instance.savefig.call_count, expected_calls)

    def test_file_reading(self):
        """Test that files can be read correctly"""
        # Test JSON reading
        with open(os.path.join(self.results_dir, 'metrics_seed0.json'), 'r') as f:
            metrics = json.load(f)
        self.assertIn('test_accuracy', metrics)
        self.assertIn('f1_score', metrics)

        # Test CSV reading
        history = pd.read_csv(os.path.join(self.results_dir, 'history_seed0.csv'))
        self.assertIn('epoch', history.columns)
        self.assertIn('accuracy', history.columns)
        self.assertIn('val_accuracy', history.columns)

        confmat = pd.read_csv(os.path.join(self.results_dir, 'confmat_seed0.csv'), index_col=0)
        self.assertEqual(confmat.shape, (2, 2))

if __name__ == '__main__':
    unittest.main()
    # اختراع شماره 140450140003002031
# مخترع نادرملکی 
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
    plt.close()

    # پردازش هر ران
    for metrics_file, hist_file, conf_file in zip(seed_metrics, seed_histories, seed_confmats):
        seed_id = os.path.splitext(os.path.basename(metrics_file))[0].split("_")[-1]

        # خواندن داده‌ها
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        history = pd.read_csv(hist_file)
        confmat = pd.read_csv(conf_file, index_col=0).values

        # تولید گزارش طبقه‌بندی
        y_true = np.repeat(range(confmat.shape[0]), confmat.sum(axis=1))
        y_pred = np.repeat(range(confmat.shape[1]), confmat.sum(axis=0))

        report_text = (
            f"نتایج Seed {seed_id}:\n"
            f"دقت: {metrics.get('test_accuracy', 'N/A'):.4f}\n"
            f"امتیاز F1: {metrics.get('f1_score', 'N/A'):.4f}\n"
            f"گزارش طبقه‌بندی:\n{classification_report(y_true, y_pred)}"
        )

        # منحنی‌های آموزش
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history['epoch'], history['accuracy'], label='دقت آموزش')
        ax.plot(history['epoch'], history['val_accuracy'], label='دقت اعتبارسنجی')
        ax.set_title(f'منحنی‌های آموزش – Seed {seed_id}')
        ax.set_xlabel('دوره')
        ax.set_ylabel('دقت')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        # ماتریس درهم‌ریختگی
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(confmat, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'ماتریس درهم‌ریختگی – Seed {seed_id}')
        pdf.savefig(fig)
        plt.close()

        # صفحه متنی
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0, 1, report_text, va='top', family='monospace', fontsize=8)
        pdf.savefig(fig)
        plt.close()

    # ---------- بخش ۲: شبیه‌سازی حسگر پیزوالکتریک ----------
    # پارامترها (مقادیر ثابت)
    PARAMS = {
        'd33': 2.5e-10,           # C/N
        'thickness_m': 0.001,     # m
        'spring_k': 1000,         # N/m
        'damping_c': 10,          # Ns/m
        'capacitance_C': 1e-6,    # Farad
        'radius': 0.01           # m
    }

    # زمان و جابجایی
    time = np.linspace(0, 1, 500)
    x = 0.01 * np.sin(2 * np.pi * 5 * time)
    dx_dt = np.gradient(x, time)

    # محاسبات
    cross_section = np.pi * PARAMS['radius'] ** 2
    F_mech = PARAMS['spring_k'] * x + PARAMS['damping_c'] * dx_dt
    sigma = F_mech / cross_section
    Q = PARAMS['d33'] * F_mech
    V = Q / PARAMS['capacitance_C']

    # رسم نمودارها
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(time, V * 1e3)
    axs[0].set_ylabel('ولتاژ (mV)')
    axs[0].set_title('ولتاژ پیزوالکتریک بر حسب زمان')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(time, F_mech)
    axs[1].set_ylabel('نیرو (N)')
    axs[1].set_title('نیروی مکانیکی بر حسب زمان')
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(time, sigma / 1e6)
    axs[2].set_ylabel('فشار (MPa)')
    axs[2].set_xlabel('زمان (s)')
    axs[2].set_title('فشار بر حسب زمان')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ---------- بخش ۳: نوآوری‌ها ----------
    innovation_text = """
نوآوری‌های مدل:#مخترع نادرملکی بدون کسب مجوز هرگونه بکارگیری موجب پیگیرد قانونی میباشد#
1. ترکیب شبکه عصبی دو نیمکره‌ای با آرایه حسگرهای پیزوالکتریک هوشمند 
   برای پردازش همزمان داده‌های مکانیکی و الکتریکی.
2. مدل فیزیکی شامل فنر، میرایی و ظرفیت خازنی برای شبیه‌سازی 
   دقیق‌تر رفتار حسگر.
3. مکانیزم گزارش‌دهی چند-ران خودکار با ترکیب نتایج یادگیری ماشین 
   و مدارهای حسگر در یک فایل PDF جامع.
"""#

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.text(0.05, 0.95, innovation_text, va='top', family='sans-serif', 
            fontsize=12, linespacing=1.5)
    pdf.savefig(fig)
    plt.close()

print(f"گزارش نهایی ساخته شد: {pdf_path}")
class FullModel(nn.Module):
    def __init__(self, input_dim, seq_len, input_proj_dim=128, hidden_size=64, 
                 num_layers=2, bidirectional=True, hemisphere_fc_size=64, 
                 lstm_dropout=0.1, fc_dropout=0.3, use_attention=False, 
                 mlp_hidden_sizes=[128, 64], n_classes=2, init_type='xavier'):
        super().__init__()

        # پیش‌پردازش / projection ورودی
        self.projector = InputProjector(input_dim, input_proj_dim, use_layernorm=True) \
            if input_dim != input_proj_dim else nn.Identity()

        # بلوک هماهنگ دو نیمکره
        self.dual = CoordinatedDualHemisphere(
            input_size=input_proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            fc_size=hemisphere_fc_size,
            lstm_dropout=lstm_dropout,
            fc_dropout=fc_dropout,
            use_attention=use_attention
        )

        # MLP برای طبقه‌بندی نهایی
        combined_size = hemisphere_fc_size * 2  # برای دو نیمکره
        mlp_layers = []
        in_features = combined_size

        for hidden_size in mlp_hidden_sizes:
            mlp_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            ])
            in_features = hidden_size

        mlp_layers.append(nn.Linear(in_features, n_classes))
        self.mlp = nn.Sequential(*mlp_layers)

        # مقداردهی اولیه وزن‌ها
        self._init_weights(init_type)

    def _init_weights(self, init_type):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_left, x_right):
        # Project input
        x_left_proj = self.projector(x_left)
        x_right_proj = self.projector(x_right)

        # Process through dual hemispheres
        features = self.dual(x_left_proj, x_right_proj)

        # Final classification
        output = self.mlp(features)

        return output
