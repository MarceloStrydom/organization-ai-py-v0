from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

def apply_dark_theme(app_or_widget):
    """Apply a modern dark theme to the application or widget"""
    
    # Define color palette
    colors = {
        'background': '#ffffff',
        'surface': '#f9fafb',
        'primary': '#3b82f6',
        'secondary': '#6b7280',
        'text': '#111827',
        'text_secondary': '#6b7280',
        'border': '#e5e7eb',
        'hover': '#f3f4f6'
    }
    
    # Create palette
    palette = QPalette()
    
    # Window colors
    palette.setColor(QPalette.ColorRole.Window, QColor(colors['background']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors['text']))
    
    # Base colors (for input fields)
    palette.setColor(QPalette.ColorRole.Base, QColor(colors['background']))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors['surface']))
    
    # Text colors
    palette.setColor(QPalette.ColorRole.Text, QColor(colors['text']))
    palette.setColor(QPalette.ColorRole.BrightText, QColor('#ffffff'))
    
    # Button colors
    palette.setColor(QPalette.ColorRole.Button, QColor(colors['surface']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors['text']))
    
    # Highlight colors
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors['primary']))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))
    
    # Apply palette
    if isinstance(app_or_widget, QApplication):
        app_or_widget.setPalette(palette)
    else:
        app_or_widget.setPalette(palette)
    
    # Global stylesheet for modern look
    stylesheet = f"""
        QWidget {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
        }}
        
        QMainWindow {{
            background-color: {colors['background']};
        }}
        
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        
        QScrollBar:vertical {{
            background-color: {colors['surface']};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {colors['border']};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {colors['secondary']};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        QLineEdit {{
            border: 1px solid {colors['border']};
            border-radius: 6px;
            padding: 8px 12px;
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        
        QLineEdit:focus {{
            border-color: {colors['primary']};
            outline: none;
        }}
        
        QTextEdit {{
            border: 1px solid {colors['border']};
            border-radius: 6px;
            padding: 8px 12px;
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        
        QTextEdit:focus {{
            border-color: {colors['primary']};
            outline: none;
        }}
        
        QComboBox {{
            border: 1px solid {colors['border']};
            border-radius: 6px;
            padding: 8px 12px;
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        
        QComboBox:focus {{
            border-color: {colors['primary']};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {colors['secondary']};
            margin-right: 5px;
        }}
        
        QSpinBox {{
            border: 1px solid {colors['border']};
            border-radius: 6px;
            padding: 8px 12px;
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        
        QSpinBox:focus {{
            border-color: {colors['primary']};
        }}
    """
    
    if isinstance(app_or_widget, QApplication):
        app_or_widget.setStyleSheet(stylesheet)
    else:
        app_or_widget.setStyleSheet(stylesheet)

def get_color_scheme():
    """Return the current color scheme"""
    return {
        'background': '#ffffff',
        'surface': '#f9fafb', 
        'primary': '#3b82f6',
        'secondary': '#6b7280',
        'text': '#111827',
        'text_secondary': '#6b7280',
        'border': '#e5e7eb',
        'hover': '#f3f4f6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'error': '#ef4444'
    }
