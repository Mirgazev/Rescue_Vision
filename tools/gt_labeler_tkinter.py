"""Minimal Tkinter bounding-box labeler skeleton for MCHS manual GT annotation."""
import json
import tkinter as tk
from pathlib import Path

class GTLabeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Rescue Vision GT Labeler')
        self.geometry('900x600')
        tk.Label(self, text='GT labeler skeleton. Add frames and export COCO annotations here.').pack(pady=30)

    def export_coco(self, path: str):
        data = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    GTLabeler().mainloop()
