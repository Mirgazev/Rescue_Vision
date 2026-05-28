#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gt_labeler_tkinter.py — простой инструмент ручной разметки bbox для
формирования ground truth в формате COCO.

Использовался в ВКР для ручной разметки 60 кадров видеоматериала МЧС
России (119 GT-боксов), результаты которой приведены в Таблице 15
ВКР (Таблица 17 в финальной нумерации).

Управление:
  ЛКМ + перетаскивание   - нарисовать bbox
  Delete / Backspace     - удалить последний bbox
  N / Right              - следующий кадр
  P / Left               - предыдущий кадр
  C                      - очистить все боксы на текущем кадре
  S                      - сохранить JSON
  Q                      - выход
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"[ERROR] требуется Pillow: {exc}")


COCO_HEADER = {
    'info': {
        'description': 'Rescue Vision manual GT annotation (MCHS video clips)',
        'version': '1.0',
        'year': 2026,
        'contributor': 'Mirgazev M. A. (MIREA, 2026)',
    },
    'licenses': [{'id': 1, 'name': 'MIT', 'url': ''}],
    'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}],
}


class GTLabeler(tk.Tk):
    def __init__(self, image_dir: str, output_json: str):
        super().__init__()
        self.title('Rescue Vision — manual GT labeler')
        self.geometry('1280x800')

        self.image_dir = Path(image_dir)
        self.output_json = Path(output_json)
        self.image_paths: List[Path] = sorted(
            list(self.image_dir.glob('*.jpg'))
            + list(self.image_dir.glob('*.png'))
            + list(self.image_dir.glob('*.jpeg'))
        )
        if not self.image_paths:
            messagebox.showerror('Error', f'No images found in {self.image_dir}')
            self.destroy()
            return

        # Состояние разметки
        self.current_index = 0
        self.annotations: Dict[str, List[Tuple[float, float, float, float]]] = {}
        self.scale = 1.0
        self.orig_size = (0, 0)
        self.draw_start: Tuple[int, int] = (0, 0)
        self.current_rect_id = None
        self.tk_image = None

        # UI
        self._build_ui()
        self._bind_keys()
        self._load_existing_annotations()
        self._show_current()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        toolbar = tk.Frame(self, bg='#222')
        toolbar.pack(side='top', fill='x')
        self.status_label = tk.Label(
            toolbar, text='', bg='#222', fg='white', font=('TkDefaultFont', 11),
        )
        self.status_label.pack(side='left', padx=10, pady=6)
        tk.Button(toolbar, text='Save (S)', command=self._save_json).pack(
            side='right', padx=4, pady=4)
        tk.Button(toolbar, text='Clear (C)', command=self._clear_current).pack(
            side='right', padx=4, pady=4)
        tk.Button(toolbar, text='Prev (P)', command=self._prev).pack(
            side='right', padx=4, pady=4)
        tk.Button(toolbar, text='Next (N)', command=self._next).pack(
            side='right', padx=4, pady=4)

        self.canvas = tk.Canvas(self, bg='#111', cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

    def _bind_keys(self) -> None:
        self.bind('<Key-n>', lambda _: self._next())
        self.bind('<Key-N>', lambda _: self._next())
        self.bind('<Right>', lambda _: self._next())
        self.bind('<Key-p>', lambda _: self._prev())
        self.bind('<Key-P>', lambda _: self._prev())
        self.bind('<Left>', lambda _: self._prev())
        self.bind('<Key-c>', lambda _: self._clear_current())
        self.bind('<Key-C>', lambda _: self._clear_current())
        self.bind('<Key-s>', lambda _: self._save_json())
        self.bind('<Key-S>', lambda _: self._save_json())
        self.bind('<Key-q>', lambda _: self.destroy())
        self.bind('<Key-Q>', lambda _: self.destroy())
        self.bind('<Delete>', lambda _: self._delete_last())
        self.bind('<BackSpace>', lambda _: self._delete_last())

    # ------------------------------------------------------------------ navigation
    def _show_current(self) -> None:
        img_path = self.image_paths[self.current_index]
        pil = Image.open(img_path)
        self.orig_size = pil.size
        canvas_w = max(self.canvas.winfo_width(), 1280)
        canvas_h = max(self.canvas.winfo_height(), 720)
        self.scale = min(canvas_w / pil.width, canvas_h / pil.height, 1.0)
        if self.scale < 1.0:
            new_size = (int(pil.width * self.scale), int(pil.height * self.scale))
            pil = pil.resize(new_size, Image.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        self._redraw_existing_boxes()
        n_boxes = len(self.annotations.get(img_path.name, []))
        self.status_label.config(
            text=(f'[{self.current_index + 1}/{len(self.image_paths)}] '
                  f'{img_path.name}   |   boxes: {n_boxes}   |   '
                  f'orig {self.orig_size[0]}×{self.orig_size[1]}   '
                  f'scale {self.scale:.2f}')
        )

    def _next(self) -> None:
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._show_current()

    def _prev(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current()

    def _clear_current(self) -> None:
        key = self.image_paths[self.current_index].name
        self.annotations[key] = []
        self._show_current()

    def _delete_last(self) -> None:
        key = self.image_paths[self.current_index].name
        boxes = self.annotations.get(key, [])
        if boxes:
            boxes.pop()
            self._show_current()

    # ------------------------------------------------------------------ drawing
    def _on_press(self, event) -> None:
        self.draw_start = (event.x, event.y)
        self.current_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline='#00ff00', width=2,
        )

    def _on_drag(self, event) -> None:
        if self.current_rect_id is None:
            return
        x0, y0 = self.draw_start
        self.canvas.coords(self.current_rect_id, x0, y0, event.x, event.y)

    def _on_release(self, event) -> None:
        if self.current_rect_id is None:
            return
        x0, y0 = self.draw_start
        x1, y1 = event.x, event.y
        self.canvas.delete(self.current_rect_id)
        self.current_rect_id = None
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return
        # Перевод в координаты оригинала
        scale = self.scale if self.scale > 0 else 1.0
        ox0, oy0 = sorted([x0, x1])[0] / scale, sorted([y0, y1])[0] / scale
        ox1, oy1 = sorted([x0, x1])[1] / scale, sorted([y0, y1])[1] / scale
        key = self.image_paths[self.current_index].name
        self.annotations.setdefault(key, []).append((ox0, oy0, ox1, oy1))
        self._show_current()

    def _redraw_existing_boxes(self) -> None:
        key = self.image_paths[self.current_index].name
        for x0, y0, x1, y1 in self.annotations.get(key, []):
            self.canvas.create_rectangle(
                x0 * self.scale, y0 * self.scale,
                x1 * self.scale, y1 * self.scale,
                outline='#00ff00', width=2,
            )

    # ------------------------------------------------------------------ IO
    def _load_existing_annotations(self) -> None:
        if not self.output_json.is_file():
            return
        try:
            with open(self.output_json, encoding='utf-8') as fh:
                coco = json.load(fh)
        except Exception:
            return
        img_id_to_name = {im['id']: im['file_name'] for im in coco.get('images', [])}
        for ann in coco.get('annotations', []):
            fn = img_id_to_name.get(ann['image_id'])
            if fn is None:
                continue
            x, y, w, h = ann['bbox']
            self.annotations.setdefault(fn, []).append((x, y, x + w, y + h))

    def _save_json(self) -> None:
        images = []
        annotations = []
        ann_id = 1
        for img_id, img_path in enumerate(self.image_paths, start=1):
            from PIL import Image as PILImage
            with PILImage.open(img_path) as im:
                w, h = im.size
            images.append({
                'id': img_id, 'file_name': img_path.name,
                'width': w, 'height': h,
            })
            for box in self.annotations.get(img_path.name, []):
                x0, y0, x1, y1 = box
                bw, bh = x1 - x0, y1 - y0
                if bw <= 0 or bh <= 0:
                    continue
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': 1,
                    'bbox': [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                    'area': round(bw * bh, 2),
                    'iscrowd': 0,
                })
                ann_id += 1
        coco = dict(COCO_HEADER)
        coco['images'] = images
        coco['annotations'] = annotations
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, 'w', encoding='utf-8') as fh:
            json.dump(coco, fh, indent=2, ensure_ascii=False)
        messagebox.showinfo(
            'Saved',
            f'Saved {len(annotations)} boxes on {len(images)} images\n'
            f'to {self.output_json}',
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True, help='Каталог с кадрами для разметки')
    p.add_argument('--output', required=True,
                   help='Путь к выходному JSON-файлу COCO')
    args = p.parse_args()
    GTLabeler(args.images, args.output).mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
