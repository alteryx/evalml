c = get_config()

c.InteractiveShellApp.exec_lines = [
    'import warnings',
    'warnings.filterwarnings('ignore')'
]
