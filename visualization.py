"""
VISUALIZATION.PY - Dashboard & Plotting
Create visualizations and save results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from config import COLORS, CONFIG
from utils import print_section, print_success

# ================================================================
# STYLE UTILITIES
# ================================================================

def styled_ax(ax):
    """Apply consistent dark theme styling to axes"""
    ax.set_facecolor(COLORS['panel'])
    ax.tick_params(colors=COLORS['subtext'], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

def create_figure():
    """Create a new figure with dark theme"""
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS['bg'])
    return fig

# ================================================================
# MAIN DASHBOARD
# ================================================================

def create_dashboard(df, pipeline, backtest_result=None):
    """
    Create comprehensive prediction dashboard
    
    Args:
        df (pd.DataFrame): Full OHLCV + features dataframe
        pipeline (dict): Output from train_full_pipeline
        backtest_result (dict): Output from run_backtest (optional)
    
    Returns:
        matplotlib.figure.Figure: Dashboard figure
    """
    print_section("📊 CREATING DASHBOARD", "📊")
    
    # Setup
    fig = create_figure()
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # Get data
    importance_df = pipeline['importance']
    metrics = pipeline['metrics']
    xgb_acc = metrics['accuracy']
    X_test = pipeline['X_test']
    y_test = pipeline['y_test']
    split_point = pipeline['split_point']
    
    # ── Panel 1: Price History ─────────────────────────────────────
    print("   Drawing price history...")
    ax1 = fig.add_subplot(gs[0, :2])
    styled_ax(ax1)
    
    price_data = df['Close'].iloc[-300:]
    ax1.plot(price_data.values, color=COLORS['blue'], linewidth=1.2)
    ax1.set_title(f'{CONFIG["TICKER"]} — Last 300 Bars',
                  color=COLORS['text'], fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylabel('Price', color=COLORS['subtext'], fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # ── Panel 2: Model Accuracy ────────────────────────────────────
    print("   Drawing accuracy...")
    ax2 = fig.add_subplot(gs[0, 2])
    styled_ax(ax2)
    
    models = ['XGBoost']
    accs = [xgb_acc * 100]
    bars = ax2.bar(models, accs, color=[COLORS['blue']], edgecolor='white', linewidth=1.5)
    ax2.set_ylim(0, 100)
    ax2.set_title('Model Accuracy', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy %', color=COLORS['subtext'], fontsize=9)
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom',
                color=COLORS['text'], fontweight='bold', fontsize=10)
    
    # ── Panel 3: Prediction Distribution ───────────────────────────
    print("   Drawing prediction distribution...")
    ax3 = fig.add_subplot(gs[1, 0])
    styled_ax(ax3)
    
    predictions = metrics['predictions']
    pred_counts = [
        (predictions == 0).sum(),
        (predictions == 1).sum()
    ]
    colors_pred = [COLORS['red'], COLORS['green']]
    ax3.bar(['DOWN', 'UP'], pred_counts, color=colors_pred, edgecolor='white', linewidth=1.5)
    ax3.set_title('Predictions', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', color=COLORS['subtext'], fontsize=9)
    
    # ── Panel 4: Win Rate ──────────────────────────────────────────
    print("   Drawing win rate...")
    ax4 = fig.add_subplot(gs[1, 1])
    styled_ax(ax4)
    
    if backtest_result and backtest_result is not None:
        win_rate = backtest_result['win_rate']
        loss_rate = 100 - win_rate
        ax4.bar(['Win %', 'Loss %'], [win_rate, loss_rate],
               color=[COLORS['green'], COLORS['red']], edgecolor='white', linewidth=1.5)
        ax4.set_ylim(0, 100)
        ax4.set_title('Backtest Win Rate', color=COLORS['text'], fontsize=11, fontweight='bold')
        ax4.set_ylabel('Percentage %', color=COLORS['subtext'], fontsize=9)
        
        ax4.text(0, win_rate + 2, f'{win_rate:.1f}%', ha='center',
                color=COLORS['green'], fontweight='bold', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No Backtest\nData', ha='center', va='center',
                transform=ax4.transAxes, color=COLORS['subtext'], fontsize=10)
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # ── Panel 5: Profit/Loss ───────────────────────────────────────
    print("   Drawing P&L...")
    ax5 = fig.add_subplot(gs[1, 2])
    styled_ax(ax5)
    
    if backtest_result and backtest_result is not None:
        total_profit = backtest_result['total_profit']
        color_pl = COLORS['green'] if total_profit > 0 else COLORS['red']
        ax5.text(0.5, 0.5, f'{total_profit:.6f}', ha='center', va='center',
                transform=ax5.transAxes, color=color_pl, fontsize=16, fontweight='bold')
        ax5.text(0.5, 0.2, 'Total P&L', ha='center',
                transform=ax5.transAxes, color=COLORS['text'], fontsize=10)
    else:
        ax5.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax5.transAxes, color=COLORS['subtext'], fontsize=10)
    
    ax5.set_xticks([])
    ax5.set_yticks([])
    
    # ── Panel 6: Feature Importance ────────────────────────────────
    print("   Drawing feature importance...")
    ax6 = fig.add_subplot(gs[2, :2])
    styled_ax(ax6)
    
    top15 = importance_df.head(15).sort_values()
    feat_c = [COLORS['green'] if any(x in f for x in ['NFP', 'Fed', 'ECB', 'Calendar'])
              else COLORS['blue'] for f in top15.index]
    ax6.barh(top15.index, top15.values, color=feat_c, edgecolor='#30363d', linewidth=0.5)
    ax6.set_title('Top 15 Features', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax6.set_xlabel('Importance', color=COLORS['subtext'], fontsize=9)
    
    # ── Panel 7: Prediction Panel ──────────────────────────────────
    print("   Drawing prediction panel...")
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(COLORS['panel'])
    for spine in ax7.spines.values():
        spine.set_edgecolor('#30363d')
    ax7.set_xticks([])
    ax7.set_yticks([])
    
    # Create prediction summary
    latest_pred = predictions[-1] if len(predictions) > 0 else 0
    direction = "📈 UP" if latest_pred == 1 else "📉 DOWN"
    
    ax7.text(0.5, 0.92, '🔮 LATEST PREDICTION',
            ha='center', transform=ax7.transAxes,
            color=COLORS['text'], fontsize=10, fontweight='bold')
    ax7.text(0.5, 0.72, CONFIG['TICKER'],
            ha='center', transform=ax7.transAxes,
            color=COLORS['subtext'], fontsize=11)
    
    color_dir = COLORS['green'] if "UP" in direction else COLORS['red']
    ax7.text(0.5, 0.55, direction,
            ha='center', transform=ax7.transAxes,
            color=color_dir, fontsize=14, fontweight='bold')
    
    ax7.text(0.5, 0.30, f'Accuracy: {xgb_acc*100:.1f}%',
            ha='center', transform=ax7.transAxes,
            color=COLORS['text'], fontsize=10)
    
    if backtest_result:
        ax7.text(0.5, 0.10, f'Win: {backtest_result["win_rate"]:.1f}%',
                ha='center', transform=ax7.transAxes,
                color=COLORS['green'], fontsize=9)
    
    print_success("Dashboard created")
    return fig

# ================================================================
# INDIVIDUAL PLOTS
# ================================================================

def plot_price_chart(ax, df, title="Price Chart"):
    """Plot price chart"""
    styled_ax(ax)
    ax.plot(df['Close'].values, color=COLORS['blue'], linewidth=1.5)
    ax.set_title(title, color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Price', color=COLORS['subtext'])
    ax.grid(True, alpha=0.2)

def plot_accuracy_comparison(ax, models_dict):
    """Plot model accuracy comparison"""
    styled_ax(ax)
    
    models = list(models_dict.keys())
    accs = [models_dict[m] * 100 for m in models]
    
    bars = ax.bar(models, accs, color=COLORS['blue'], edgecolor='white', linewidth=1.5)
    ax.set_ylim(0, 100)
    ax.set_title('Model Accuracy', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy %', color=COLORS['subtext'])
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{acc:.1f}%', ha='center', va='bottom', color=COLORS['text'])

def plot_feature_importance(ax, importance_series, top_n=15):
    """Plot feature importance"""
    styled_ax(ax)
    
    top_features = importance_series.head(top_n).sort_values()
    feat_colors = [COLORS['green'] if any(x in f for x in ['NFP', 'Fed', 'ECB'])
                   else COLORS['blue'] for f in top_features.index]
    
    ax.barh(top_features.index, top_features.values, color=feat_colors)
    ax.set_title(f'Top {top_n} Features', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Importance', color=COLORS['subtext'])

# ================================================================
# SAVE & DISPLAY
# ================================================================

def save_dashboard(fig, filename=None):
    """Save dashboard to file"""
    if filename is None:
        filename = 'prediction_dashboard.png'
    
    print_section("💾 SAVING DASHBOARD", "💾")
    
    try:
        fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
        print_success(f"Dashboard saved as '{filename}'")
        return True
    except Exception as e:
        print(f"   ❌ Error saving dashboard: {e}")
        return False

def show_dashboard(fig):
    """Display dashboard"""
    try:
        plt.show()
        return True
    except Exception as e:
        print(f"   ⚠️  Could not display dashboard: {e}")
        return False

def export_results(pipeline, backtest_result=None):
    """Export results to CSV"""
    print_section("📁 EXPORTING RESULTS", "📁")
    
    # Export feature importance
    pipeline['importance'].to_csv('feature_importance.csv')
    print_success("Feature importance saved to 'feature_importance.csv'")
    
    # Export backtest results if available
    if backtest_result and 'trades' in backtest_result:
        backtest_result['trades'].to_csv('backtest_trades.csv', index=False)
        print_success("Backtest trades saved to 'backtest_trades.csv'")
